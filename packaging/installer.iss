; Inno Setup script. Build with packaging/build.ps1 (needs dist\Meld from PyInstaller first).
; Installs per user (no administrator prompt) to %LOCALAPPDATA%\Programs\Meld.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{71DFA20E-4FE8-4CA1-9CEB-5677B92E97DA}
AppName=Meld
AppVersion={#AppVersion}
AppPublisher=Meld
DefaultDirName={autopf}\Meld
DefaultGroupName=Meld
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
SetupIconFile=meld.ico
UninstallDisplayIcon={app}\Meld.exe
InfoBeforeFile=NOTICE.txt
OutputDir=..\dist-installer
OutputBaseFilename=Meld-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a shortcut on the desktop"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\Meld\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Meld"; Filename: "{app}\Meld.exe"
Name: "{autodesktop}\Meld"; Filename: "{app}\Meld.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Meld.exe"; Description: "Start Meld now"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; the updated YouTube downloader and saved choices; the videos you made (in Documents) are never touched
Type: filesandordirs; Name: "{localappdata}\Meld"
