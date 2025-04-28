import json
import difflib
from pipeline.unified_inference import ContextAwareOCRPipeline

def load_ground_truth(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def evaluate_fields(predicted, ground_truth):
    scores = {}
    total_score = 0
    for key in ground_truth:
        gt = ground_truth.get(key, "").strip().lower()
        pred = predicted.get(key, "").strip().lower()
        ratio = difflib.SequenceMatcher(None, gt, pred).ratio()
        scores[key] = round(ratio, 2)
        total_score += ratio
    average_score = round(total_score / len(ground_truth), 2)
    return scores, average_score

def main(image_path, ground_truth_json):
    pipeline = ContextAwareOCRPipeline()
    result = pipeline.process_image(image_path)
    
    # Parse the structured output from LLM
    try:
        predicted_json = json.loads(result["structured_output"])
    except:
        print("❌ LLM output was not valid JSON")
        print(result["structured_output"])
        return

    gt = load_ground_truth(ground_truth_json)
    field_scores, avg_score = evaluate_fields(predicted_json, gt)

    print("📊 Evaluation Results:")
    for field, score in field_scores.items():
        print(f"{field}: {score * 100:.2f}% match")
    print(f"\n✅ Average Accuracy: {avg_score * 100:.2f}%")

if __name__ == "__main__":
    main("samples/sample_card.jpg", "samples/sample_card_groundtruth.json")
