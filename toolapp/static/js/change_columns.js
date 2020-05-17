function MoveFeatureLeft()
{
    let selected = $('#id_features option:selected');
    selected.appendTo('#id_columns');
}

function MoveFeatureRight()
{
    let selected = $('#id_columns option:selected');
    selected.appendTo('#id_features');
}

function MoveLeft(id)
{
    let value = document.getElementById(id).value;
    document.getElementById(id).value = '';
    document.getElementById('id_columns').options.add(new Option(value, value));
}

function MoveRight(id)
{
    let selected = $("#id_columns option:selected");
    let value = document.getElementById(id).value;
    if (selected.length > 0) {
      if (selected.length > 1)
        alert("Выбраны несколько столбцов. Вы можете выбрать только один столбец.");
      else {
        document.getElementById("id_columns").options.remove(document.getElementById("id_columns").options.selectedIndex);
        if (value !== "")
          document.getElementById('id_columns').options.add(new Option(value, value));
        document.getElementById(id).value = selected[0].value;
      }
    }
}
function selectAll() { 
        selectBox = document.getElementById("id_features");

        for (let i = 0; i < selectBox.options.length; i++) 
        { 
             selectBox.options[i].selected = true; 
        } 
}