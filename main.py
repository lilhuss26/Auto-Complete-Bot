from src.Pipeline import Pipeline

pipeline = Pipeline()
suggestion = pipeline.get_suggestions("i was about")
print(suggestion)

