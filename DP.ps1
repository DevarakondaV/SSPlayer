# File deletes contents of log and runs program

$path = "C:\Users\Vishnu\Documents\EngProj\tflog"
$dir_contents = Get-ChildItem -Path $path
if ($dir_contents.Length -gt 0) {
    Write-Output "Deleteing Contentes"
    $dir_contents = $path+"\"+$dir_contents
    foreach ($v in $dir_contents){
        Remove-Item -Path $v
    }
}


$python_path = "C:\Users\vishnu\Envs\RL\scripts\python.exe"
$prog_path = "C:\Users\vishnu\Documents\EngProj\SSPlayer\tf.py"

.$python_path $prog_path

