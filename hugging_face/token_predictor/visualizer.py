from dotenv import load_dotenv
from token_predictor import TokenPredictor
from token_graph import create_token_graph, visualize_predictions

load_dotenv(override=True)

message = "In one sentence, describe the color orange to someone who has never been able to see"
model_name = "gpt-4.1-mini"

predictor = TokenPredictor(model_name)
predictions = predictor.predict_tokens(message)
G = create_token_graph(model_name, predictions)
plt = visualize_predictions(G)
plt.show()
