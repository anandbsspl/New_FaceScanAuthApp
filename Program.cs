using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Speech.Synthesis;
using System.Text.Json;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using DlibDotNet;
using DlibDotNet.Extensions;
using FaceRecognitionDotNet;
using Point = DlibDotNet.Point;
using Rectangle = DlibDotNet.Rectangle;

class FaceAuthSystem
{
    // Configuration
    const double EAR_THRESHOLD = 0.21;
    const int REQUIRED_BLINKS = 3;
    const int REQUIRED_HEAD_MOVES = 2;
    const double HEAD_MOVE_THRESHOLD = 25.0;
    const double BASE_THRESHOLD = 0.65;
    const int MIN_FACE_SIZE = 100;
    const int MAX_CAPTURE_SECONDS = 30;
    const int REQUIRED_SAMPLES = 3;

    static string usersFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Users");
    static string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models");

    static FaceRecognition faceRecognition;
    static ShapePredictor shapePredictor;
    static SpeechSynthesizer synthesizer = new SpeechSynthesizer();
    static VideoCapture camera;
    static string cameraWindowName = "Face Capture";

    class UserData
    {
        public string Name { get; set; }
        public List<double[]> Embeddings { get; set; } = new List<double[]>();
        public List<bool> HasGlasses { get; set; } = new List<bool>();
        public List<bool> HasFacialHair { get; set; } = new List<bool>();
        public DateTime LastUpdated { get; set; }
    }

    static void Main()
    {
        try
        {
            InitializeSystem();
            RunMainMenu();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Fatal error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
        finally
        {
            CleanupResources();
        }
    }

    static void InitializeSystem()
    {
        Console.WriteLine("Initializing system...");
        Directory.CreateDirectory(usersFolder);
        Directory.CreateDirectory(modelPath);

        Console.WriteLine("Loading models...");
        faceRecognition = FaceRecognition.Create(modelPath);
        shapePredictor = ShapePredictor.Deserialize(Path.Combine(modelPath, "shape_predictor_68_face_landmarks.dat"));

        // Initialize camera
        try
        {
            camera = new VideoCapture(0, VideoCapture.API.DShow);
            if (!camera.IsOpened)
            {
                throw new Exception("Camera initialization failed");
            }
            camera.Set(CapProp.FrameWidth, 1280);
            camera.Set(CapProp.FrameHeight, 720);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Camera error: {ex.Message}");
            Environment.Exit(1);
        }
    }

    static void CleanupResources()
    {
        faceRecognition?.Dispose();
        synthesizer?.Dispose();
        camera?.Stop();  // Add this line
        camera?.Dispose();
        CvInvoke.DestroyAllWindows();
    }

    static void RunMainMenu()
    {
        try
        {
            while (true)
            {
                Console.Clear();
                Console.WriteLine("==== FaceScanAuthApp ====");
                Console.WriteLine("1. Register User");
                Console.WriteLine("2. Authenticate User");
                Console.WriteLine("3. List Users");
                Console.WriteLine("4. Delete User");
                Console.WriteLine("5. Exit");
                Console.Write("Select an option: ");

                switch (Console.ReadLine())
                {
                    case "1": RegisterUser(); break;
                    case "2": AuthenticateUser(); break;
                    case "3": ListUsers(); break;
                    case "4": DeleteUser(); break;
                    case "5": return;
                    default: Console.WriteLine("Invalid option"); break;
                }
                Console.WriteLine("\nPress any key to continue...");
                Console.ReadKey();
            }
        }
        finally
        {
            CleanupResources();
        }
    }

    static void RegisterUser()
    {
        Console.Write("Enter name: ");
        string name = Console.ReadLine()?.Trim();

        if (string.IsNullOrEmpty(name))
        {
            Console.WriteLine("Name cannot be empty!");
            return;
        }

        string userFile = Path.Combine(usersFolder, $"{name}.json");
        if (File.Exists(userFile))
        {
            Console.WriteLine("Username already exists!");
            Speak("Username already exists");
            return;
        }

        Console.WriteLine("\nWe'll capture your face as you naturally appear:");
        var (encodings, hasGlasses, hasFacialHair) = CaptureFaceSamples(REQUIRED_SAMPLES);
        if (encodings == null || encodings.Count < REQUIRED_SAMPLES)
        {
            Console.WriteLine("Registration failed - couldn't capture enough samples");
            Speak("Authentication failed couldn't capture face properly");
            return;
        }

        if (CheckExistingUser(encodings.First(), name))
        {
            return;
        }

        SaveUserProfile(name, encodings, hasGlasses, hasFacialHair);
        Console.WriteLine("✅ Registration successful!");
        synthesizer.SpeakAsync("Registration successful");
        Speak("Registration successful");
    }

    static void AuthenticateUser()
    {
        Console.WriteLine("\nPlease face the camera naturally");
        var authData = CaptureFaceSamples(REQUIRED_SAMPLES);
        if (authData.encodings == null || authData.encodings.Count < REQUIRED_SAMPLES)
        {
            Console.WriteLine("Authentication failed - couldn't capture face properly");
            Speak("Authentication failed couldn't capture face properly");
            return;
        }

        var users = LoadAllUsers();
        string authenticatedUser = null;
        double highestConfidence = 0;
        var similarityResults = new List<string>();

        foreach (var user in users)
        {
            double minSimilarity = 1.0;
            bool allPassed = true;
            var userSimilarities = new List<double>();

            for (int i = 0; i < authData.encodings.Count; i++)
            {
                double bestSimilarity = 0;
                for (int j = 0; j < user.Embeddings.Count; j++)
                {
                    double similarity = 1 - FaceDistance(
                        user.Embeddings[j],
                        authData.encodings[i].GetRawEncoding().ToArray());

                    double threshold = GetDynamicThreshold(
                        authData.hasGlasses[i], authData.hasFacialHair[i],
                        user.HasGlasses[j], user.HasFacialHair[j]);

                    if (similarity > bestSimilarity)
                        bestSimilarity = similarity;

                    if (similarity >= threshold)
                        break;
                }

                userSimilarities.Add(bestSimilarity);
                similarityResults.Add($"Sample {i + 1} vs {user.Name}: {bestSimilarity:P1}");

                if (bestSimilarity < BASE_THRESHOLD * 0.9)
                {
                    allPassed = false;
                    break;
                }

                minSimilarity = Math.Min(minSimilarity, bestSimilarity);
            }

            if (allPassed && minSimilarity > highestConfidence)
            {
                highestConfidence = minSimilarity;
                authenticatedUser = user.Name;
            }
        }

        // Display all similarity results
        Console.WriteLine("\nSimilarity Results:");
        foreach (var result in similarityResults)
        {
            Console.WriteLine(result);
        }

        if (authenticatedUser != null)
        {
            Console.WriteLine($"\n✅ Welcome {authenticatedUser}! (Confidence: {highestConfidence:P0})");
            synthesizer.SpeakAsync($"Welcome back {authenticatedUser}");
            Speak($"Welcome back {authenticatedUser}");
        }
        else
        {
            Console.WriteLine("\n❌ Authentication failed");
            synthesizer.SpeakAsync("Authentication failed");
            Speak("Authentication failed");
        }
    }

    static double GetDynamicThreshold(bool authGlasses, bool authHair, bool regGlasses, bool regHair)
    {
        // More lenient when appearances match
        if (authGlasses == regGlasses && authHair == regHair)
            return BASE_THRESHOLD;

        // Slightly more strict for partial matches
        if (authGlasses == regGlasses || authHair == regHair)
            return BASE_THRESHOLD * 0.95;

        // Most strict for completely different appearances
        return BASE_THRESHOLD * 0.9;
    }

    static (List<FaceEncoding> encodings, List<bool> hasGlasses, List<bool> hasFacialHair) CaptureFaceSamples(int requiredSamples)
    {
        var encodings = new List<FaceEncoding>();
        var hasGlasses = new List<bool>();
        var hasFacialHair = new List<bool>();

        try
        {
            CvInvoke.NamedWindow(cameraWindowName, WindowFlags.KeepRatio);

            DateTime startTime = DateTime.Now;
            int blinkCount = 0, headMoveCount = 0;
            Point? lastNosePos = null;
            var earHistory = new List<double>();
            string currentInstruction = "Please look directly at the camera";
            System.Drawing.Rectangle focusRect = new System.Drawing.Rectangle();
            bool faceInPosition = false;

            while ((DateTime.Now - startTime).TotalSeconds < MAX_CAPTURE_SECONDS &&
                   encodings.Count < requiredSamples)
            {
                using (var frame = camera.QueryFrame())
                {
                    if (frame == null) continue;

                    using (var image = frame.ToImage<Bgr, byte>())
                    {
                        // Detect face and analyze
                        var faceData = DetectAndAnalyzeFace(image);

                        // Update focus rectangle and position status
                        if (faceData.HasValue)
                        {
                            var faceRect = faceData.Value.shape.Rect;
                            // Explicitly convert uint to int for rectangle coordinates
                            focusRect = new System.Drawing.Rectangle(
                                (int)faceRect.Left,
                                (int)faceRect.Top,
                                (int)faceRect.Width,
                                (int)faceRect.Height);

                            // Check if face is in good position
                            faceInPosition = IsFaceInGoodPosition(image.Width, image.Height, focusRect);
                        }
                        else
                        {
                            faceInPosition = false;
                        }

                        // Rest of the method remains the same...
                        // Update instructions based on state
                        if (encodings.Count > 0)
                        {
                            currentInstruction = $"Captured {encodings.Count}/{requiredSamples} samples";
                        }
                        else if (!faceData.HasValue)
                        {
                            currentInstruction = "Please position your face in the frame";
                        }
                        else if (!faceInPosition)
                        {
                            currentInstruction = "Move slightly to center your face";
                        }
                        else if (blinkCount < REQUIRED_BLINKS)
                        {
                            currentInstruction = "Blink naturally 3 times";
                        }
                        else if (headMoveCount < REQUIRED_HEAD_MOVES)
                        {
                            currentInstruction = "Slowly turn your head side to side";
                        }

                        // Draw UI elements
                        DrawCaptureUI(image, focusRect, faceInPosition, currentInstruction,
                                     blinkCount, REQUIRED_BLINKS,
                                     headMoveCount, REQUIRED_HEAD_MOVES,
                                     encodings.Count, requiredSamples);

                        // Process face data if available
                        if (faceData.HasValue && faceInPosition)
                        {
                            UpdateLivenessCounters(faceData.Value.shape, ref blinkCount, ref headMoveCount,
                                                 ref earHistory, ref lastNosePos);

                            if (blinkCount >= REQUIRED_BLINKS && headMoveCount >= REQUIRED_HEAD_MOVES)
                            {
                                encodings.Add(faceData.Value.encoding);
                                hasGlasses.Add(faceData.Value.hasGlasses);
                                hasFacialHair.Add(faceData.Value.hasFacialHair);

                                Speak($"Captured sample {encodings.Count}");

                                // Reset for next sample
                                blinkCount = 0;
                                headMoveCount = 0;
                                earHistory.Clear();
                                lastNosePos = null;
                            }
                        }

                        CvInvoke.WaitKey(10);
                    }
                }
            }
        }
        finally
        {
            CvInvoke.DestroyWindow(cameraWindowName);
        }

        return (encodings.Count >= requiredSamples ? encodings : null, hasGlasses, hasFacialHair);
    }

    static void DrawCaptureUI(Image<Bgr, byte> image,
                         System.Drawing.Rectangle faceRect,
                         bool faceInPosition,
                         string instruction,
                         int currentBlinks, int requiredBlinks,
                         int currentHeadMoves, int requiredHeadMoves,
                         int currentSamples, int totalSamples)
    {
        int width = image.Width;
        int height = image.Height;

        // Draw focus indicator (green if good position, red otherwise)
        var focusColor = faceInPosition ?
            new MCvScalar(0, 255, 0) : // Green
            new MCvScalar(0, 0, 255);  // Red

        // Convert rectangle coordinates to int explicitly
        int rectX = (int)faceRect.X;
        int rectY = (int)faceRect.Y;
        int rectWidth = (int)faceRect.Width;
        int rectHeight = (int)faceRect.Height;

        // Draw face rectangle with thickness based on position quality
        CvInvoke.Rectangle(image,
                          new System.Drawing.Rectangle(rectX, rectY, rectWidth, rectHeight),
                          focusColor,
                          faceInPosition ? 2 : 1);

        // Draw center guidelines
        CvInvoke.Line(image,
                     new System.Drawing.Point(width / 2, 0),
                     new System.Drawing.Point(width / 2, height),
                     new MCvScalar(0, 255, 255), 1);
        CvInvoke.Line(image,
                     new System.Drawing.Point(0, height / 2),
                     new System.Drawing.Point(width, height / 2),
                     new MCvScalar(0, 255, 255), 1);

        // Draw instruction text
        CvInvoke.PutText(image, instruction,
                        new System.Drawing.Point(10, 30),
                        FontFace.HersheySimplex, 0.8,
                        new MCvScalar(255, 255, 255), 2);

        // Draw progress bars
        DrawProgressBar(image, 40, "Blink Progress", currentBlinks, requiredBlinks);
        DrawProgressBar(image, 70, "Head Move Progress", currentHeadMoves, requiredHeadMoves);
        DrawProgressBar(image, 100, "Sample Progress", currentSamples, totalSamples);

        CvInvoke.Imshow(cameraWindowName, image);
    }

    static void DrawProgressBar(Image<Bgr, byte> image, int yPos, string label,
                              int current, int max)
    {
        int width = 200;
        int height = 20;
        int xPos = 10;

        // Background
        CvInvoke.Rectangle(image,
                          new System.Drawing.Rectangle(xPos, yPos, width, height),
                          new MCvScalar(100, 100, 100), -1);

        // Progress
        if (max > 0)
        {
            int progressWidth = (int)(width * ((double)current / max));
            CvInvoke.Rectangle(image,
                              new System.Drawing.Rectangle(xPos, yPos, progressWidth, height),
                              new MCvScalar(0, 200, 0), -1);
        }

        // Border
        CvInvoke.Rectangle(image,
                          new System.Drawing.Rectangle(xPos, yPos, width, height),
                          new MCvScalar(200, 200, 200), 1);

        // Text
        CvInvoke.PutText(image, $"{label}: {current}/{max}",
                        new System.Drawing.Point(xPos + width + 10, yPos + 15),
                        FontFace.HersheySimplex, 0.5,
                        new MCvScalar(255, 255, 255), 1);
    }

    static bool IsFaceInGoodPosition(int frameWidth, int frameHeight, System.Drawing.Rectangle faceRect)
    {
        // Check if face is centered and properly sized
        double centerTolerance = 0.2;
        double sizeTolerance = 0.3;

        // Convert to int explicitly
        int faceX = (int)faceRect.X;
        int faceWidth = (int)faceRect.Width;
        int faceY = (int)faceRect.Y;
        int faceHeight = (int)faceRect.Height;

        double centerX = faceX + (faceWidth / 2.0);
        double centerY = faceY + (faceHeight / 2.0);

        bool xCentered = Math.Abs(centerX - (frameWidth / 2.0)) < (frameWidth * centerTolerance);
        bool yCentered = Math.Abs(centerY - (frameHeight / 2.0)) < (frameHeight * centerTolerance);

        bool goodWidth = faceWidth > (frameWidth * (0.2 - sizeTolerance)) &&
                        faceWidth < (frameWidth * (0.4 + sizeTolerance));
        bool goodHeight = faceHeight > (frameHeight * (0.2 - sizeTolerance)) &&
                         faceHeight < (frameHeight * (0.4 + sizeTolerance));

        return xCentered && yCentered && goodWidth && goodHeight;
    }

    static void Speak(string message)
    {
        try
        {
            synthesizer.SpeakAsyncCancelAll();  // Cancel any pending speech
            synthesizer.Speak(message);  // Use synchronous speak for important messages
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Voice error: {ex.Message}");
        }
    }

    static void DisplayStatus(Image<Bgr, byte> image, int blinks, int headMoves, int currentSample, int totalSamples)
    {
        CvInvoke.PutText(image, $"Sample: {currentSample}/{totalSamples}", new System.Drawing.Point(10, 30),
                        FontFace.HersheySimplex, 0.7, new MCvScalar(0, 255, 255), 2);
        CvInvoke.PutText(image, $"Blinks: {blinks}/{REQUIRED_BLINKS}", new System.Drawing.Point(10, 60),
                        FontFace.HersheySimplex, 0.7, new MCvScalar(0, 255, 0), 2);
        CvInvoke.PutText(image, $"Head moves: {headMoves}/{REQUIRED_HEAD_MOVES}", new System.Drawing.Point(10, 90),
                        FontFace.HersheySimplex, 0.7, new MCvScalar(0, 255, 0), 2);
        CvInvoke.Imshow(cameraWindowName, image);
    }

    static (FaceEncoding encoding, FullObjectDetection shape, bool hasGlasses, bool hasFacialHair)?
        DetectAndAnalyzeFace(Image<Bgr, byte> image)
    {
        using (var bitmap = image.ToBitmap())
        using (var dlibImg = bitmap.ToArray2D<RgbPixel>())
        {
            var faces = Dlib.GetFrontalFaceDetector().Operator(dlibImg);
            if (faces.Length == 0) return null;

            var face = faces[0];
            if (face.Width < MIN_FACE_SIZE || face.Height < MIN_FACE_SIZE)
            {
                ShowInstruction(image, "Move closer", Color.Red);
                return null;
            }

            var shape = shapePredictor.Detect(dlibImg, face);
            bool glasses = DetectGlasses(shape);
            bool facialHair = DetectFacialHair(shape);

            string tempFile = Path.GetTempFileName();
            try
            {
                bitmap.Save(tempFile, System.Drawing.Imaging.ImageFormat.Jpeg);
                using var frImage = FaceRecognition.LoadImageFile(tempFile);
                var encoding = faceRecognition.FaceEncodings(frImage).FirstOrDefault();
                return (encoding, shape, glasses, facialHair);
            }
            finally
            {
                File.Delete(tempFile);
            }
        }
    }

    static bool DetectGlasses(FullObjectDetection shape)
    {
        var leftEye = new[] { shape.GetPart(36), shape.GetPart(37), shape.GetPart(38),
                             shape.GetPart(39), shape.GetPart(40), shape.GetPart(41) };
        var rightEye = new[] { shape.GetPart(42), shape.GetPart(43), shape.GetPart(44),
                              shape.GetPart(45), shape.GetPart(46), shape.GetPart(47) };

        return (CalculateEAR(leftEye) < EAR_THRESHOLD * 0.7 ||
                CalculateEAR(rightEye) < EAR_THRESHOLD * 0.7);
    }

    static bool DetectFacialHair(FullObjectDetection shape)
    {
        var jawPoints = Enumerable.Range(0, 17).Select(i => shape.GetPart((uint)i)).ToArray();
        double jawWidth = Distance(jawPoints[0], jawPoints[16]);
        double chinHeight = Distance(jawPoints[8], shape.GetPart(33)); // Chin to nose tip

        return (jawWidth / chinHeight) > 4.2;
    }

    static double CalculateEAR(IEnumerable<Point> eyePoints)
    {
        var points = eyePoints.ToArray();
        double a = Distance(points[1], points[5]);
        double b = Distance(points[2], points[4]);
        double c = Distance(points[0], points[3]);
        return (a + b) / (2.0 * c);
    }

    static void UpdateLivenessCounters(FullObjectDetection shape, ref int blinkCount, ref int headMoveCount,
                                     ref List<double> earHistory, ref Point? lastNosePos)
    {
        double leftEAR = CalculateEAR(Enumerable.Range(36, 6).Select(i => shape.GetPart((uint)i)));
        double rightEAR = CalculateEAR(Enumerable.Range(42, 6).Select(i => shape.GetPart((uint)i)));
        double avgEAR = (leftEAR + rightEAR) / 2.0;

        earHistory.Add(avgEAR);
        if (earHistory.Count > 3 && avgEAR < EAR_THRESHOLD && earHistory[^2] >= EAR_THRESHOLD)
        {
            blinkCount++;
        }

        var nose = shape.GetPart(30);
        if (lastNosePos.HasValue)
        {
            double xDiff = Math.Abs(nose.X - lastNosePos.Value.X);
            double yDiff = Math.Abs(nose.Y - lastNosePos.Value.Y);
            if ((xDiff > HEAD_MOVE_THRESHOLD || yDiff > HEAD_MOVE_THRESHOLD) && earHistory.Count > 5)
            {
                headMoveCount++;
            }
        }
        lastNosePos = nose;
    }

    static bool CheckExistingUser(FaceEncoding newEncoding, string currentUsername)
    {
        foreach (var file in Directory.GetFiles(usersFolder, "*.json"))
        {
            var user = JsonSerializer.Deserialize<UserData>(File.ReadAllText(file));
            if (user.Name == currentUsername) continue;

            foreach (var existing in user.Embeddings)
            {
                double similarity = 1 - FaceDistance(existing, newEncoding.GetRawEncoding().ToArray());
                if (similarity > BASE_THRESHOLD)
                {
                    Console.WriteLine($"❌ User already registered as: {user.Name} (Similarity: {similarity:P0})");
                    synthesizer.SpeakAsync("This face is already registered");
                    Speak("This face is already registered");
                    return true;
                }
            }
        }
        return false;
    }

    static void SaveUserProfile(string name, List<FaceEncoding> encodings,
                              List<bool> hasGlasses, List<bool> hasFacialHair)
    {
        var newUser = new UserData
        {
            Name = name,
            LastUpdated = DateTime.Now,
            Embeddings = encodings.Select(e => e.GetRawEncoding().ToArray()).ToList(),
            HasGlasses = hasGlasses,
            HasFacialHair = hasFacialHair
        };

        File.WriteAllText(Path.Combine(usersFolder, $"{name}.json"),
                         JsonSerializer.Serialize(newUser));
    }

    static List<UserData> LoadAllUsers()
    {
        return Directory.GetFiles(usersFolder, "*.json")
            .Select(f => JsonSerializer.Deserialize<UserData>(File.ReadAllText(f)))
            .ToList();
    }

    static void ListUsers()
    {
        var users = LoadAllUsers();
        if (!users.Any())
        {
            Console.WriteLine("No registered users found.");
            return;
        }

        Console.WriteLine("\nRegistered Users:");
        Console.WriteLine("----------------");
        foreach (var user in users)
        {
            Console.WriteLine($"- {user.Name} (Last updated: {user.LastUpdated})");
            Console.WriteLine($"  Samples: {user.Embeddings.Count}");
            Console.WriteLine($"  Glasses Samples: {user.HasGlasses.Count(g => g)}");
            Console.WriteLine($"  Facial Hair Samples: {user.HasFacialHair.Count(f => f)}");
        }
    }

    static void DeleteUser()
    {
        var users = LoadAllUsers();
        if (!users.Any())
        {
            Console.WriteLine("No users to delete.");
            return;
        }

        Console.WriteLine("\nSelect user to delete:");
        for (int i = 0; i < users.Count; i++)
        {
            Console.WriteLine($"{i + 1}. {users[i].Name}");
        }

        Console.Write("\nEnter user number to delete (0 to cancel): ");
        if (int.TryParse(Console.ReadLine(), out int choice) && choice > 0 && choice <= users.Count)
        {
            var user = users[choice - 1];
            Console.Write($"Delete {user.Name}? (y/n): ");
            if (Console.ReadLine().ToLower() == "y")
            {
                File.Delete(Path.Combine(usersFolder, $"{user.Name}.json"));
                Console.WriteLine($"User {user.Name} deleted.");
                synthesizer.SpeakAsync("User deleted");
                Speak("User deleted");
            }
        }
    }

    static double Distance(Point p1, Point p2)
        => Math.Sqrt(Math.Pow(p1.X - p2.X, 2) + Math.Pow(p1.Y - p2.Y, 2));

    static double FaceDistance(double[] enc1, double[] enc2)
    {
        if (enc1 == null || enc2 == null || enc1.Length != 128 || enc2.Length != 128)
            return double.MaxValue;
        return Math.Sqrt(enc1.Zip(enc2, (a, b) => Math.Pow(a - b, 2)).Sum());
    }

    static void ShowInstruction(Image<Bgr, byte> image, string text, Color color)
    {
        CvInvoke.PutText(image, text, new System.Drawing.Point(10, 30),
                    FontFace.HersheySimplex, 0.7, new MCvScalar(color.B, color.G, color.R), 2);
        CvInvoke.Imshow(cameraWindowName, image);
        CvInvoke.WaitKey(1);
    }
}