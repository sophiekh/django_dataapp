let data_table = document.querySelector('table');
data_table.className = "table table-hover table-sm";
if (classification) {
  let predictionRow = document.querySelectorAll('td:nth-child(2)');
  for (let cell of predictionRow) {
    if (cell.innerHTML != 'None') {
      cell.parentElement.className = "prediction";
    }
  }  
}