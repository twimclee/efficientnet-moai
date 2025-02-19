import json

def train_ready_callback():
    message = {
        "status": "preparing",
        "message": "학습 환경을 준비중입니다.",
    }

    print(json.dumps(message, ensure_ascii=False), flush=True)

def train_start_callback():
    message = {
        "status": "start",
        "message": "모델 학습을 시작합니다."
    }

    print(json.dumps(message, ensure_ascii=False), flush=True)

def train_epoch_end_callback(epoch, remaining, epoch_acc, epoch_loss):
    log_data = {
        "epoch": epoch,
        "time": remaining,
        "accuracy": epoch_acc,
        "loss": epoch_loss,
    }

    message = {
        "status": "in_progress",
        "message": log_data
    }

    print(json.dumps(message, ensure_ascii=False), flush=True)

def train_end_callback():
    message = {
        "status": "complete",
        "message": "모델 학습이 성공적으로 완료되었습니다."
    }

    print(json.dumps(message, ensure_ascii=False), flush=True)

def inference_ready_callback():
    message = {
        "status": "preparing",
        "message": "추론 환경을 준비중입니다.",
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

def inference_start_callback():
    message = {
        "status": "start",
        "message": "모델 추론을 시작합니다."
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

def inference_epoch_end_callback(current_epoch, total_epochs):
    message = {
        "status": "in_progress",
        "message": f"이미지 처리 중... ({current_epoch}/{total_epochs})",
        "progress": {
            "current": current_epoch,
            "total": total_epochs
        }
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

def inference_end_callback():
    message = {
        "status": "complete",
        "message": "모델 추론이 완료되었습니다."
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

def export_ready_callback():
    message = {
        "status": "preparing",
        "message": "모델 내보내기 준비중입니다."
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

def export_start_callback():
    message = {
        "status": "start",
        "message": "모델 내보내기 시작"
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)

def export_end_callback():
    message = {
        "status": "complete",
        "message": "모델 내보내기가 완료되었습니다."
    }
    print(json.dumps(message, ensure_ascii=False), flush=True)