Chart.defaults.global.defaultFontColor = 'black';
  Chart.defaults.global.defaultFontFamily = 'Helvetica Neue';
  let chart_data = [];
  $.ajax({
    method: "GET",
    url: endpoint1,
    success: function(data){
      chart_data = JSON.parse(data);
      drawChart(chart_data.standard, 'scatter', 'Без предварительного нормирования');
      drawChart(chart_data.normalized, "scatter_normalized", "С нормированием");
    },
    error: function(error_data){
      console.log("error");
      console.log(error_data);
    }
  })
  function drawChart(data, id, label ) {
    let plot1 = document.getElementById(id);
    colors = ['#FF0000', '#006400', '#000080', 
     '#800000',  '#00FFFF', '#FFFF00', '#000000', '#FF00FF'] 
    let chartData = data;
    for (let i = 0; i < data.datasets.length; i++) {
      let color = (i < colors.length) ? colors[i] : ("#"+((1<<24)*Math.random()|0).toString(16)); 
      chartData.datasets[i].backgroundColor = color;
    }
    let scatterChart = new Chart(plot1, {
      type: 'scatter',
      data: chartData,
      options: {
          scales: {
              xAxes: [{
                  type: 'linear',
                  position: 'bottom'
              }]
          },
          title: {
            display: true,
            text: label,
            fontSize: 18,
            position: 'bottom',
            fontStyle: 'normal'
        }
      }
    });
    
  }