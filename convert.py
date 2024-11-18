import pandas as pd
from pandas import DataFrame
import json
import sys
import os


def process_ground_truth(sheet_data):
    # Extracting the main columns (ground truth) like time, X, Y, Z
    ground_truth_columns = ["time,s", "X,m", "Y,m", "Z,m"]
    ground_truth_data = sheet_data.loc[:, ground_truth_columns]
    return ground_truth_data


def process_settings(sheet_data: DataFrame):
    # Process camera settings and matrix data from the 'Settings' part
    settings = {
        "focal_length_mm": sheet_data.iat[1, 7],
        "sensor_width_mm": sheet_data.iat[2, 7],
        "sensor_height_mm": sheet_data.iat[3, 7],
    }

    return settings


def process_cameras(sheet_data: DataFrame):
    # Process positions for 3 cameras
    all_camera_settings = process_settings(sheet_data)
    cameras = {}
    # row number - 1 - header row
    cameras_indicies = [5, 12, 19]
    x_shift = [2, 1]
    y_shift = [3, 1]
    z_shift = [4, 1]
    azimuth_shift = [5, 1]

    # Find the indices for each camera's position data
    for cam_number, cam_sheet_row in enumerate(cameras_indicies):
        cam_sheet_index = {"row": cam_sheet_row, "column": 6}
        camera = f"camera{cam_number}"
        cameras[camera] = {
            "position": {
                "x_m": sheet_data.iat[
                    cam_sheet_index["row"] + x_shift[0],
                    cam_sheet_index["column"] + x_shift[1],
                ],
                "y_m": sheet_data.iat[
                    cam_sheet_index["row"] + y_shift[0],
                    cam_sheet_index["column"] + y_shift[1],
                ],
                "z_m": sheet_data.iat[
                    cam_sheet_index["row"] + z_shift[0],
                    cam_sheet_index["column"] + z_shift[1],
                ],
                "azimuth_deg": sheet_data.iat[
                    cam_sheet_index["row"] + azimuth_shift[0],
                    cam_sheet_index["column"] + azimuth_shift[1],
                ],
            },
            "matrix": {
                "focal_length_mm": all_camera_settings["focal_length_mm"],
                "sensor_width_mm": all_camera_settings["sensor_width_mm"],
                "sensor_height_mm": all_camera_settings["sensor_height_mm"],
            },
        }

    return cameras


def convert_to_csv(sheet_data: DataFrame, output_file):
    # Process ground truth and save it to CSV
    ground_truth_data = process_ground_truth(sheet_data)
    ground_truth_data.to_csv(output_file, index=False)
    print(f"Ground truth data has been saved to {output_file}")


def convert_to_json(sheet_data: DataFrame, output_file):
    # Gather all data to export as JSON
    cameras = process_cameras(sheet_data)
    ground_truth_data = process_ground_truth(sheet_data).to_dict(orient="records")

    # Combine into a single dictionary
    data = {
        "ground_truth": ground_truth_data,
        "cameras": cameras,
        "object": {"object_diameter_m": sheet_data.iat[27, 7]},
    }

    # Writing to JSON
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Data has been saved to {output_file}")

    return cameras


if __name__ == "__main__":
    # Load the sheet (assuming data is in the first sheet)
    # parse input file from first argument

    excel_file = sys.argv[1]
    if excel_file == "all":
        for excel_file in [
            "data/step1/videoset1/Seq1_settings.xlsx",
            "data/step1/videoset2/Seq2_settings.xlsx",
            "data/step1/videoset3/Seq3_settings.xlsx",
            "data/step1/videoset4/Seq4_settings.xlsx",
            "data/step1/videoset5/Seq5_settings.xlsx",
            "data/step1/videoset6/Seq6_settings.xlsx",
            "data/step1/videoset7/Seq7_settings.xlsx",
            "data/step1/videoset8/Seq8_settings.xlsx",
        ]:
            sheet_data: DataFrame = pd.read_excel(excel_file)

            # Convert to CSV and JSON
            output_file_name = os.path.splitext(excel_file)[0]
            convert_to_csv(sheet_data, f"{output_file_name}.csv")
            convert_to_json(sheet_data, f"{output_file_name}.json")
