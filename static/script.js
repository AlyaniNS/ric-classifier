const ctx = document.getElementById('similarityChart').getContext('2d');
let chart = new Chart(ctx, {
    type: 'bar',
    data: { labels: [], datasets: [{
        label: 'Similarity (%)',
        data: [],
        backgroundColor: [],
        borderColor: [],
        borderWidth: 1
    }] },
    options: {
        indexAxis: 'y', // <-- makes the bar chart horizontal
        plugins: {
            legend: { display: false },
            datalabels: {
                anchor: 'end',
                align: 'right',
                formatter: (value) => value.toFixed(1) + '%'
            }
        },
        scales: { x: { beginAtZero: true, max: 100 } },
        animation: { duration: 800 }
    },
    plugins: [ChartDataLabels]
});


document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Show preview
    const preview = document.getElementById('preview');
    preview.src = URL.createObjectURL(fileInput.files[0]);
    preview.style.display = 'block';

    const response = await fetch('/predict', { method: 'POST', body: formData });
    const result = await response.json();
    if (result.error) { alert(result.error); return; }

    document.getElementById('prediction').innerText = "Predicted Class: " + result.predicted_class;

    const colors = Object.keys(result.similarities).map((_, idx) =>
        idx === result.predicted_idx ? 'rgba(255, 165, 0, 0.8)' : 'rgba(54, 162, 235, 0.6)'
    );

    chart.data.labels = Object.keys(result.similarities);
    chart.data.datasets[0].data = Object.values(result.similarities);
    chart.data.datasets[0].backgroundColor = colors;
    chart.data.datasets[0].borderColor = colors.map(c => c.replace('0.6', '1'));
    chart.update();
});
