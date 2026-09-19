from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_files_path:str
    test_file_path:str