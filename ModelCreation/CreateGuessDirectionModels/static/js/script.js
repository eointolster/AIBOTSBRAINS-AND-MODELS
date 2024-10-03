document.addEventListener('DOMContentLoaded', function() {
    // 2D Grid Setup
    const gridCanvas = document.getElementById('gridCanvas');
    const ctx = gridCanvas.getContext('2d');
    const gridSize = 28;
    const cellSize = gridCanvas.width / gridSize;

    let startPos = { x: 0, y: 0 };
    let endPos = { x: 27, y: 27 };
    let faceAngle = 0;

    // 2D Chart Setup
    const chartCtx = document.getElementById('lossChart').getContext('2d');
    const lossChart = new Chart(chartCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '2D Training Loss',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // 3D Chart Setup
    const chart3DCtx = document.getElementById('loss3DChart').getContext('2d');
    const loss3DChart = new Chart(chart3DCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '3D Training Loss',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    function drawGrid() {
        ctx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);
        
        // Draw grid lines
        ctx.strokeStyle = '#ccc';
        for (let i = 0; i <= gridSize; i++) {
            ctx.beginPath();
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, gridCanvas.height);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo(gridCanvas.width, i * cellSize);
            ctx.stroke();
        }

        // Draw start position (red square with face)
        ctx.fillStyle = 'red';
        ctx.fillRect(startPos.x * cellSize, startPos.y * cellSize, cellSize, cellSize);
        
        // Draw face (triangle)
        ctx.save();
        ctx.translate((startPos.x + 0.5) * cellSize, (startPos.y + 0.5) * cellSize);
        ctx.rotate(faceAngle);
        ctx.beginPath();
        ctx.moveTo(0, -cellSize/4);
        ctx.lineTo(cellSize/4, cellSize/4);
        ctx.lineTo(-cellSize/4, cellSize/4);
        ctx.closePath();
        ctx.fillStyle = 'yellow';
        ctx.fill();
        ctx.restore();

        // Draw end position (green square)
        ctx.fillStyle = 'green';
        ctx.fillRect(endPos.x * cellSize, endPos.y * cellSize, cellSize, cellSize);
    }

    function generateRandomPositions() {
        startPos = {
            x: Math.floor(Math.random() * gridSize),
            y: Math.floor(Math.random() * gridSize)
        };
        endPos = {
            x: Math.floor(Math.random() * gridSize),
            y: Math.floor(Math.random() * gridSize)
        };
        faceAngle = Math.random() * 2 * Math.PI;
        drawGrid();
    }

    drawGrid();

    let trainingInterval;
    let training3DInterval;

    // 2D Model Controls
    document.getElementById('startTrainBtn').addEventListener('click', function() {
        fetch('/start_training', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                trainingInterval = setInterval(updateTrainingStatus, 100);
            });
    });

    document.getElementById('stopTrainBtn').addEventListener('click', function() {
        fetch('/stop_training', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                clearInterval(trainingInterval);
            });
    });

    function updateTrainingStatus() {
        fetch('/get_training_status')
            .then(response => response.json())
            .then(data => {
                document.getElementById('trainingIterations').textContent = data.iterations;
                document.getElementById('currentLoss').textContent = data.loss.toFixed(4);
                
                lossChart.data.labels.push(data.iterations);
                lossChart.data.datasets[0].data.push(data.loss);
                lossChart.update();

                if (data.iterations % 10 === 0) {  // Update positions every 10 iterations
                    generateRandomPositions();
                }

                if (!data.is_training) {
                    clearInterval(trainingInterval);
                }
            });
    }

    document.getElementById('predictBtn').addEventListener('click', function() {
        generateRandomPositions();
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                positions: [startPos.x, startPos.y, endPos.x, endPos.y] 
            }),
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('predictedAngle').textContent = data.predicted_angle.toFixed(2);
            document.getElementById('correctAngle').textContent = data.correct_angle.toFixed(2);
            document.getElementById('isCorrect').textContent = data.is_correct;
        });
    });

    document.getElementById('saveModelBtn').addEventListener('click', function() {
        fetch('/save_model', { method: 'POST' })
            .then(response => response.json())
            .then(data => console.log(data.message));
    });

    // 3D Model Controls
    document.getElementById('start3DTrainBtn').addEventListener('click', function() {
        fetch('/start_3d_training', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                training3DInterval = setInterval(update3DTrainingStatus, 100);
            });
    });

    document.getElementById('stop3DTrainBtn').addEventListener('click', function() {
        fetch('/stop_3d_training', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(data.message);
                clearInterval(training3DInterval);
            });
    });

    function update3DTrainingStatus() {
        fetch('/get_3d_training_status')
            .then(response => response.json())
            .then(data => {
                document.getElementById('training3DIterations').textContent = data.iterations;
                document.getElementById('current3DLoss').textContent = data.loss.toFixed(4);
                
                loss3DChart.data.labels.push(data.iterations);
                loss3DChart.data.datasets[0].data.push(data.loss);
                loss3DChart.update();

                if (!data.is_training) {
                    clearInterval(training3DInterval);
                }
            });
    }

    document.getElementById('predict3DBtn').addEventListener('click', function() {
        const positions = Array(6).fill().map(() => Math.random() * 2 - 1); // Range: -1 to 1
        fetch('/predict_3d', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ positions: positions }),
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('predicted3DDirection').textContent = `(${data.predicted_direction.map(v => v.toFixed(4)).join(', ')})`;
            document.getElementById('correct3DDirection').textContent = `(${data.correct_direction.map(v => v.toFixed(4)).join(', ')})`;
            document.getElementById('is3DCorrect').textContent = data.is_correct;
            document.getElementById('angleDifference').textContent = data.angle_difference.toFixed(2) + '°';
        });
    });

    document.getElementById('save3DModelBtn').addEventListener('click', function() {
        fetch('/save_3d_model', { method: 'POST' })
            .then(response => response.json())
            .then(data => console.log(data.message));
    });

    document.getElementById('load3DModelBtn').addEventListener('click', function() {
        fetch('/load_3d_model', { method: 'POST' })
            .then(response => response.json())
            .then(data => console.log(data.message));
    });
});