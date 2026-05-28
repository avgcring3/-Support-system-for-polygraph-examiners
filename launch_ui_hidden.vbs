Option Explicit

Dim fso, shell, projectRoot, cmd, i
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)

If IsServerUp("http://127.0.0.1:8501") Then
  shell.Run "http://127.0.0.1:8501", 1, False
  WScript.Quit 0
End If

cmd = "cmd /c cd /d """ & projectRoot & """ && python -m streamlit run ui\streamlit_app.py --server.port 8501 --server.headless true"
shell.Run cmd, 0, False

For i = 1 To 30
  WScript.Sleep 500
  If IsServerUp("http://127.0.0.1:8501") Then Exit For
Next

shell.Run "http://127.0.0.1:8501", 1, False

Function IsServerUp(url)
  On Error Resume Next
  Dim xhr
  Set xhr = CreateObject("MSXML2.XMLHTTP")
  xhr.Open "GET", url, False
  xhr.Send
  If Err.Number = 0 Then
    IsServerUp = (xhr.Status >= 200 And xhr.Status < 500)
  Else
    IsServerUp = False
    Err.Clear
  End If
  On Error GoTo 0
End Function
