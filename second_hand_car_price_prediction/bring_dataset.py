import gdown
import pathlib
    # Option 1: Using file ID
    
output_path=pathlib.Path("data")
output_path=output_path / 'raw'
output_path.mkdir(parents=True,exist_ok=True)

file_id = '10XN9ugutKXcoUzZhqztRFsGNrwR_JlD8'
gdown.download(id=file_id, output='./data/raw/car_details.csv', quiet=False)