const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;

    canvas.addEventListener('mousedown', () => isDrawing = true);
    canvas.addEventListener('mouseup', () => isDrawing = false);
    canvas.addEventListener('mousemove', draw);

    function draw(event) {
        if (!isDrawing) return;
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 10;
        ctx.lineCap = 'round';
        ctx.lineTo(event.offsetX, event.offsetY);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(event.offsetX, event.offsetY);
    }

    function goback(e) {
        e.preventDefault();
        window.history.back();
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
    }

    function uploadimg(){
        document.querySelector('.drawingbox').style.display = 'none';
        document.querySelector('form').style.display = 'block';
        document.querySelectorAll('.btns')[0].style.display = 'none';
        document.querySelectorAll('.btns')[1].style.display = 'block';
        document.querySelector('.prediction_area').style.display = 'flex';

    }

    function drawimg(){
        document.querySelector('.drawingbox').style.display = 'flex';
        document.querySelector('form').style.display = 'none';
        document.querySelectorAll('.btns')[1].style.display = 'none';
        document.querySelectorAll('.btns')[0].style.display = 'block';
        document.querySelector('.prediction_area').style.display = 'flex';
    }

    function predictFromCanvas() {
        // Convert canvas content to a data URL
        canvas.toBlob((blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'digit.png'); // Send as 'file' to match /predict route

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                const predictionResult = document.getElementById('predictionResult');
                const predictedClass = data.predicted_class;
                const probabilities = data.probabilities;
                const barchart = document.querySelector('.barchart');
                
                barchart.style.display = 'block';

                // Display the predicted digit
                predictionResult.innerText = "Predicted Digit: " + predictedClass;

                // Update the bars based on probabilities
                probabilities.forEach((probability, index) => {
                    const bar = document.querySelector(`.bar${index + 1}`);
                    const heightPercentage = probability * 100;
                    
                    bar.style.height = `${heightPercentage}%`;
                    bar.style.backgroundColor = index === predictedClass ? '#0066cc' : 'gray';
                    bar.innerText = `${(probability * 100).toFixed(4)}%`;
                });
            })
            .catch(error => {
                document.getElementById('predictionResult').innerText = "Error in prediction. Please try again.";
            });
        });
    }




        function submitForm(event) {
        event.preventDefault();

        var formData = new FormData(document.getElementById('predictForm'));
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            const predictionResult = document.getElementById('predictionResult');
            const predictedClass = data.predicted_class;
            const probabilities = data.probabilities;
            const barchart = document.querySelector('.barchart');
            const btn = document.querySelector('.btn-custom');
            barchart.style.display = 'block';

            // Display the predicted digit
            predictionResult.innerText = "Predicted Digit: " + predictedClass;

            // Update the bars based on probabilities
            probabilities.forEach((probability, index) => {
                const bar = document.querySelector(`.bar${index + 1}`);
                const heightPercentage = probability * 100; // Convert to percentage for height
                
                bar.style.height = `${heightPercentage}%`;
                bar.style.backgroundColor = index === predictedClass ? '#0066cc' : 'gray'; // Highlight predicted class
                bar.innerText = `${(probability * 100).toFixed(4)}%`; // Show percentage
            });
        })
        .catch(error => {
            document.getElementById('predictionResult').innerText = "Error in prediction. Please try again.";
        });
    }