function updateFileInfo() {
    var fileInput = document.getElementById('x-ray-upload');
    var fileName = fileInput.files[0].name;
    document.getElementById('file-name').textContent = fileName;
    document.getElementById('file-info').style.display = 'block';
    document.getElementById('selected-file-name').textContent = 'File: ' + fileName;
}

function removeFile() {
    var fileInput = document.getElementById('x-ray-upload');
    fileInput.value = '';
    document.getElementById('file-info').style.display = 'none';
    document.getElementById('file-name').textContent = 'Choose X-ray image';
}
