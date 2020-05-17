let input = document.getElementById("id_data");
  let label = document.getElementById('output');
  input.addEventListener('change', function(e){
    let fileName = e.target.value.split( '\\' ).pop();
    if( fileName )
      label.innerHTML = fileName;
	});