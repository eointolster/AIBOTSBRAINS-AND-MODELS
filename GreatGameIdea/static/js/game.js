let scene, camera, renderer, human, aiBots = [], projectiles = [], goal;
let gameState = {};
let gameLoop;
let controls; // For zoom and pan controls

function initThree() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x87CEEB);  // Sky blue background

    const gameBoard = document.getElementById('game-board');
    const width = gameBoard.clientWidth;
    const height = gameBoard.clientHeight;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    gameBoard.appendChild(renderer.domElement);

    const aspect = width / height;

    camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    camera.position.set(0, 20, 30);
    camera.lookAt(0, 0, 0);

    // Check if OrbitControls are available
    if (typeof THREE.OrbitControls !== 'undefined') {
        // Add OrbitControls for zoom and pan
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        controls.screenSpacePanning = false;
        controls.maxPolarAngle = Math.PI / 2;
    } else {
        console.warn('OrbitControls not available. Zoom and pan features will be disabled.');
    }

    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(10, 20, 10);
    light.castShadow = true;
    scene.add(light);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    createBoard();
}

function createTextSprite(text) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = 'bold 32px Arial'; // Smaller font size
    context.fillStyle = 'rgba(0, 0, 0, 1)';
    
    // Measure text width and height
    const textMetrics = context.measureText(text);
    const textWidth = textMetrics.width;
    const textHeight = 32; // Approximate height based on font size
    
    // Set canvas size to fit rotated text
    canvas.width = textHeight;
    canvas.height = textWidth;
    
    // Rotate and draw text
    context.translate(canvas.width / 2, canvas.height / 2);
    context.rotate(Math.PI / 2);
    context.fillText(text, -textWidth / 2, textHeight / 4);
    
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter; // Prevents blurry text
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);
    
    // Adjust sprite scale
    sprite.scale.set(0.5, 0.5, 1);
    
    return sprite;
}

function createBoard() {
    const boardGroup = new THREE.Group();

    const planeGeometry = new THREE.PlaneGeometry(30, 20);
    const planeMaterial = new THREE.MeshPhongMaterial({ color: 0x2ecc71 });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    plane.rotation.x = -Math.PI / 2;
    plane.receiveShadow = true;
    boardGroup.add(plane);

    const xOffset = (30 - 1) / 2;
    const yOffset = (20 - 1) / 2;

    for (let x = 0; x < 30; x++) {
        for (let y = 0; y < 20; y++) {
            if ((x + y) % 2 === 0) continue;

            const squareGeometry = new THREE.BoxGeometry(1, 0.1, 1);
            const squareMaterial = new THREE.MeshPhongMaterial({ color: 0x27ae60 });
            const square = new THREE.Mesh(squareGeometry, squareMaterial);
            square.position.set(x - xOffset, 0.05, y - yOffset);
            square.receiveShadow = true;
            boardGroup.add(square);

            // Add label
            const label = `${String.fromCharCode(65 + y)}${x + 1}`;
            const sprite = createTextSprite(label);
            sprite.position.set(x - xOffset, 0.06, y - yOffset); // Slightly above the board
            sprite.rotation.x = -Math.PI / 2; // Rotate to lie flat
            boardGroup.add(sprite);
        }
    }

    scene.add(boardGroup);
}

function createHuman() {
    const geometry = new THREE.SphereGeometry(0.4, 32, 32);
    const material = new THREE.MeshPhongMaterial({ color: 0x3498db });
    human = new THREE.Mesh(geometry, material);
    human.castShadow = true;
    scene.add(human);
    updateHumanPosition();
}

function createAIBots() {
    const xOffset = (30 - 1) / 2;
    const yOffset = (20 - 1) / 2;
    gameState.ai_bots.forEach(bot => {
        const botGroup = new THREE.Group();

        // Create the bot sphere
        const geometry = new THREE.SphereGeometry(0.4, 32, 32);
        const material = new THREE.MeshPhongMaterial({ color: 0xe74c3c });
        const aiBot = new THREE.Mesh(geometry, material);
        aiBot.castShadow = true;
        botGroup.add(aiBot);

        // Create the direction cone
        const coneGeometry = new THREE.ConeGeometry(0.2, 0.4, 32);
        const coneMaterial = new THREE.MeshPhongMaterial({ color: 0xf1c40f });
        const cone = new THREE.Mesh(coneGeometry, coneMaterial);
        cone.position.set(0, 0, 0.4);  // Position the cone in front of the sphere
        cone.rotation.x = Math.PI / 2;  // Rotate to point forward
        botGroup.add(cone);

        botGroup.position.set(bot.x - xOffset, 0.5, bot.y - yOffset);
        scene.add(botGroup);
        aiBots.push(botGroup);
    });
}


function createGoal() {
    const geometry = new THREE.BoxGeometry(0.8, 0.8, 0.8);
    const material = new THREE.MeshPhongMaterial({ color: 0xf1c40f });
    goal = new THREE.Mesh(geometry, material);
    const xOffset = (30 - 1) / 2;
    const yOffset = (20 - 1) / 2;
    goal.position.set(gameState.goal.x - xOffset, 0.5, gameState.goal.y - yOffset);
    goal.castShadow = true;
    scene.add(goal);
}

function updateHumanPosition() {
    const xOffset = (30 - 1) / 2;
    const yOffset = (20 - 1) / 2;
    human.position.set(gameState.human.x - xOffset, 0.5, gameState.human.y - yOffset);
}

function updateAIBots() {
    const xOffset = (30 - 1) / 2;
    const yOffset = (20 - 1) / 2;
    aiBots.forEach((botGroup, index) => {
        if (index < gameState.ai_bots.length) {
            const bot = gameState.ai_bots[index];
            botGroup.position.set(bot.x - xOffset, 0.5, bot.y - yOffset);
            
            // Update the rotation based on the facing direction
            switch(bot.facing) {
                case 'up':
                    botGroup.rotation.y = Math.PI;
                    break;
                case 'down':
                    botGroup.rotation.y = 0;
                    break;
                case 'left':
                    botGroup.rotation.y = -Math.PI / 2;
                    break;
                case 'right':
                    botGroup.rotation.y = Math.PI / 2;
                    break;
            }
        } else {
            scene.remove(botGroup);
        }
    });
    aiBots = aiBots.slice(0, gameState.ai_bots.length);
}

function updateProjectiles() {
    projectiles.forEach(p => scene.remove(p));
    projectiles = [];
    const xOffset = (30 - 1) / 2;
    const yOffset = (20 - 1) / 2;
    gameState.projectiles.forEach(p => {
        const geometry = new THREE.SphereGeometry(0.2, 8, 8);
        const material = new THREE.MeshLambertMaterial({ color: 0xffffff });
        const projectile = new THREE.Mesh(geometry, material);
        projectile.position.set(p.x - xOffset, 0.5, p.y - yOffset);
        scene.add(projectile);
        projectiles.push(projectile);
    });
}

function animate() {
    requestAnimationFrame(animate);
    if (controls) {
        controls.update(); // Update controls in the animation loop
    }
    renderer.render(scene, camera);
}

function updateGameState() {
    fetch('/get_game_state')
        .then(response => response.json())
        .then(data => {
            gameState = data;
            updateHumanPosition();
            updateAIBots();
            updateProjectiles();
            checkGameOver();
            // No need to update labels as they are static
        });
}

document.getElementById('start-game').addEventListener('click', () => {
    const numBots = document.getElementById('num-bots').value;
    const aiMode = document.getElementById('ai-mode').value;
    
    // First, set the AI mode
    fetch('/set_ai_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: aiMode })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Now start the game
            fetch('/start_game', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ num_bots: parseInt(numBots) })
            })
            .then(response => response.json())
            .then(data => {
                gameState = data;
                document.getElementById('start-menu').style.display = 'none';
                initThree();
                createHuman();
                createAIBots();
                createGoal();
                animate();
                gameLoop = setInterval(updateGameState, 100);  // Update game state every 100ms
            });
        }
    });
});

document.addEventListener('keydown', (event) => {
    if (!gameState.game_over) {
        let direction;
        switch(event.key) {
            case 'ArrowUp': direction = 'up'; break;
            case 'ArrowDown': direction = 'down'; break;
            case 'ArrowLeft': direction = 'left'; break;
            case 'ArrowRight': direction = 'right'; break;
            default: return;
        }
        fetch('/move_human', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ direction: direction })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                gameState.human[direction === 'left' || direction === 'right' ? 'x' : 'y'] += 
                    (direction === 'up' || direction === 'left') ? -1 : 1;
                updateHumanPosition();
            }
        });
    }
});

function checkGameOver() {
    if (gameState.game_over) {
        clearInterval(gameLoop);
        document.getElementById('game-over').style.display = 'block';
        document.getElementById('winner-message').textContent = 
            gameState.winner === 'human' ? 'You win!' : 'Game over! You lost.';
    }
}

document.getElementById('restart-game').addEventListener('click', () => {
    location.reload();
});
