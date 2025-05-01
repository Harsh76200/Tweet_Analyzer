// Change theme between light and dark mode
document.addEventListener('DOMContentLoaded', function() {
    const themeSwitch = document.getElementById('theme-switch');
    
    if (themeSwitch) {
        themeSwitch.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
            
            // Update switch text and icon
            const switchSpan = themeSwitch.querySelector('span');
            if (document.body.classList.contains('dark-mode')) {
                switchSpan.textContent = 'Light Mode';
                themeSwitch.querySelector('i').classList.replace('fa-moon', 'fa-sun');
            } else {
                switchSpan.textContent = 'Dark Mode';
                themeSwitch.querySelector('i').classList.replace('fa-sun', 'fa-moon');
            }
            
            // Optional: Save preference in localStorage
            localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
        });
        
        // Check for saved preference
        if (localStorage.getItem('darkMode') === 'true') {
            document.body.classList.add('dark-mode');
            themeSwitch.querySelector('span').textContent = 'Light Mode';
            themeSwitch.querySelector('i').classList.replace('fa-moon', 'fa-sun');
        }
    }
});

// Initialize charts with sentiment data
function initializeCharts(data) {
    const isDarkMode = document.body.classList.contains('dark-mode');
    const textColor = isDarkMode ? '#ffffff' : '#14171A';
    
    // Set Chart.js defaults suitable for both themes
    Chart.defaults.color = textColor;
    Chart.defaults.borderColor = isDarkMode ? '#38444d' : '#E1E8ED';
    
    // Create sentiment pie chart
    if (document.getElementById('sentimentPieChart') || document.getElementById('landingSentimentPieChart')) {
        const ctx = document.getElementById('sentimentPieChart') || document.getElementById('landingSentimentPieChart');
        
        new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [
                        data.sentiment_counts.positive,
                        data.sentiment_counts.neutral,
                        data.sentiment_counts.negative
                    ],
                    backgroundColor: [
                        '#4CAF50',  // Positive - green
                        '#757575',  // Neutral - gray
                        '#F44336'   // Negative - red
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = Math.round((value / total) * 100);
                                return `${label}: ${percentage}% (${value})`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Create candidate bar chart
    if (document.getElementById('candidateBarChart') || document.getElementById('landingCandidateBarChart')) {
        const ctx = document.getElementById('candidateBarChart') || document.getElementById('landingCandidateBarChart');
        const candidates = Object.keys(data.candidate_sentiment);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: candidates,
                datasets: [
                    {
                        label: 'Positive',
                        data: candidates.map(c => data.candidate_sentiment[c].positive),
                        backgroundColor: '#4CAF50',
                        barPercentage: 0.8,
                        categoryPercentage: 0.7
                    },
                    {
                        label: 'Neutral',
                        data: candidates.map(c => data.candidate_sentiment[c].neutral),
                        backgroundColor: '#757575',
                        barPercentage: 0.8,
                        categoryPercentage: 0.7
                    },
                    {
                        label: 'Negative',
                        data: candidates.map(c => data.candidate_sentiment[c].negative),
                        backgroundColor: '#F44336',
                        barPercentage: 0.8,
                        categoryPercentage: 0.7
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        stacked: false,
                        grid: {
                            display: false
                        }
                    },
                    y: {
                        stacked: false,
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.dataset.label || '';
                                const value = context.raw || 0;
                                return `${label}: ${value}%`;
                            }
                        }
                    }
                }
            }
        });
    }
}

// Update charts when theme changes
document.getElementById('theme-switch')?.addEventListener('click', function() {
    if (sentimentData) {
        // A small delay to ensure DOM has updated with new theme
        setTimeout(() => initializeCharts(sentimentData), 50);
    }
});