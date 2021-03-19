class SAScsv(object):
    """
    SUMMARY:    convert SAS7bdat files in a dir  --> .csv files
    INPUT :     path to the file
    METHODS:    .cp_to_child_folder    <--copy your .CSV output files
                                       to a child "/csv/" folder


    NOTE:       Files in the path should are recognised as long as
                the filename contains the character: "."
                Modify the filenames accordingly if needed.
    """

    def __init__(self, path):
        self.converted = 0
        self.path = path
        self.list = self.getNames()
        self.dirtocsv()
        self.list = self.getNames()

    def dirtocsv(self):
        self.go2dir(self.path)
        import pandas as pd
        print("Converting .SAS7bdat files to .csv")

        def filetocsv(file):
            self.go2dir(self.path)
            df = pd.read_sas(file, encoding='latin_1')
            nn = file.split('.')[0] + '.csv'
            df.to_csv(nn, sep=',', index=None, encoding='latin_1')
            self.increment_converted()
            print(f"Currently converted: {self.converted} files")

        [filetocsv(f) for f in self.list]
        print("Conversion:  <<< Done >>>")

    def cp_to_child_folder(self):
        print("Moving your output files to /csv sub-folder")
        self.csv_path = self.csvPath()
        self.move2csvFolder()
        self.cleanParentFolder()
        print("Move:    <<< Done >>>")

    def getNames(self):
        import os
        os.chdir(self.path)
        return os.listdir()

    def move2csvFolder(self):
        from shutil import copy2
        self.go2dir(self.path)
        [copy2(f, self.csv_path) for f in self.list if '.csv' in f]

    def cleanParentFolder(self):
        import os
        [os.remove(f) for f in self.list if '.csv' in f]

    def csvPath(self):
        from pathlib import Path
        Path(f"{self.path}/csv").mkdir(parents=True, exist_ok=True)
        np = Path(f"{self.path}/csv")
        linux_np = str(np)
        windows_np = str(np.as_posix())
        if self.getOS():
            return windows_np
        else:
            return linux_np

    def go2dir(self, pPath):
        import os
        os.chdir(pPath)

    def getOS(self):
        from sys import platform
        print(" --- Analyzing your OS ---")
        if platform.startswith("lin"):
            # linux
            print("Your OS type is: " + platform)
            x = False
        elif platform.startswith("dar"):
            # OS X
            print("Your OS type is: " + platform)
            x = False
        elif platform.startswith("win"):
            # Windows
            print("Your OS type is: " + platform)
            x = True
        return x

    def increment_converted(self):
        self.converted += 1
