import os
import json
from inference_sdk import InferenceHTTPClient

# Roboflow API 設定
RF_API_URL = "https://detect.roboflow.com"
RF_API_KEY = "jPtq6WNi8tOUWrx5we1t"
WORKSPACE_NAME = "eli-uxy4p"
WORKFLOW_ID = "custom-workflow"

client = InferenceHTTPClient(api_url=RF_API_URL, api_key=RF_API_KEY)

def get_roboflow_predictions(image_path):
    """
    調用 Roboflow API，傳入圖片進行偵測，返回預測結果
    """
    try:
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image": image_path},
            use_cache=True
        )

        # 解析預測結果
        raw_preds = result[0]["predictions"]["predictions"]
        clean_preds = [
            {"x": pred["x"], "y": pred["y"], "width": pred["width"], "height": pred["height"]}
            for pred in raw_preds
        ]
        return {"predictions": clean_preds}

    except Exception as e:
        print(f"Roboflow API 調用失敗: {e}")
        return {"predictions": []}
