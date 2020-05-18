let data_table = document.querySelectorAll('table');
for (let table of data_table){
  table.className = "table table-hover table-sm";
}
if (classification) {
  let predictionRow = document.querySelectorAll('div.data-table td:nth-child(2)');
  for (let cell of predictionRow) {
    if (cell.innerHTML != 'None') {
      cell.parentElement.className = "prediction";
    }
  }

}



