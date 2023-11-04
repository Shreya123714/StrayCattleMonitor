// Function to load live webcam feed
function loadWebcamFeed() {
    const videoElement = document.getElementById('video-element');
    navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
            videoElement.srcObject = stream;
        })
        .catch((error) => {
            console.error('Error accessing webcam:', error);
        });
}

// Execute when the page loads
window.addEventListener('load', () => {
    loadWebcamFeed();
});

// Function to draw bounding boxes on detected objects
function drawBoundingBoxes(objects) {
    const canvas = document.getElementById('detection-canvas');
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);

    for (const obj of objects) {
        const { class_name, confidence, x, y, width, height } = obj;
        context.strokeStyle = 'green';
        context.lineWidth = 2;
        context.fillStyle = 'green';
        context.font = '16px Arial';

        context.beginPath();
        context.rect(x, y, width, height);
        context.stroke();
        context.fillText(`${class_name} (${Math.round(confidence * 100)}%)`, x, y - 5);
    }
}

// Function to perform real-time detection
function detectObjects() {
    const videoElement = document.getElementById('video-element');
    const canvas = document.getElementById('detection-canvas');
    const net = new cvstfjs.Model();
    net.loadModel('model.json');
    net.initiate();
    const classes = ['class1', 'class2', 'class3']; // Define your object classes

    videoElement.addEventListener('play', () => {
        setInterval(async () => {
            const image = cvstfjs.browser.fromPixels(videoElement);
            const predictions = await net.predict(image);
            const objects = predictions.map((prediction) => {
                const [x, y, width, height] = prediction.bbox;
                const class_name = classes[prediction.class];
                const confidence = prediction.score;
                return { class_name, confidence, x, y, width, height };
            });
            drawBoundingBoxes(objects);
        }, 1000 / 10); // Adjust the interval as needed
    });
}

// Execute when the page loads
window.addEventListener('load', () => {
    loadWebcamFeed();
    detectObjects();
});
