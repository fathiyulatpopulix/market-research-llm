# %%
import os
import time
from openai import OpenAI

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# %%
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define price points and other parameters
prices = [749, 999, 1249, 1499]
n_samples = 20
model = "gpt-4.1"
# %%
# Collect all data
class TokenTracker:
    def __init__(self):
        self.data = []

    def _get_model_entry(self, model):
        for entry in self.data:
            if entry['model'] == model:
                return entry
        # If not found, initialize
        new_entry = {'model': model, 'prompt': 0, 'completion': 0}
        self.data.append(new_entry)
        return new_entry

    def add_usage(self, model, prompt_tokens=0, completion_tokens=0):
        entry = self._get_model_entry(model)
        entry['prompt'] += prompt_tokens
        entry['completion'] += completion_tokens

    def get_data(self):
        return self.data
# %%
from dataclasses import dataclass, asdict

@dataclass
class SurveyDataEntry:
    price: float = 0.0
    response: str = ""
    decision: bool = False
# %%
# Prompt template
prompt_template = (
    "A customer is randomly selected while shopping for laptops. Their annual income is $70000.\n"
    "While shopping, the customer sees a Surface Laptop 3, Price: ${}, Processor: Intel Core i5, "
    "RAM: 8GB, Screen Size: 13.5in, SD: 128GB\n"
    "The customer is asked, after they finish shopping: Did you purchase any laptop? If so, which one?\n"
    "Customer:"
)

# Function to generate responses
def generate_responses(price: float, n: int, model: str, token_tracker: TokenTracker) -> list[SurveyDataEntry]:
    responses = []
    # sys_prompt = system_prompt.format(price)
    prompt = prompt_template.format(price)
    for _ in range(n):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Act as text completion model."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1.0,
                max_tokens=50,
            )
            token_tracker.add_usage(
                model,
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
            answer = completion.choices[0].message.content.strip().lower()
            responses.append(SurveyDataEntry(price=price, response=answer))
            time.sleep(0.1)  # To avoid hitting rate limits
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    return responses
# %%
survey_data: list[SurveyDataEntry] = []
token_tracker = TokenTracker()

for price in prices:
    print(f"Collecting responses at price ${price}")
    survey_data.extend(generate_responses(price, n_samples, model, token_tracker))
# %%
# # Print results
# for d in survey_data:
#     print(f"\nPrice: ${d.price}")
#     print(f" > {d.response}")
# %%
from pydantic import BaseModel
class BinaryAnswer(BaseModel):
    decision: bool

system_prompt_template = (
    "Map the user answer into yes/no based on question below\n"
    "Question: {}"
)

# %%
# Run GPT with structured response format
evaluator_model = "gpt-4o-mini"
def binarize_answers(answers_list: list[SurveyDataEntry], binary_question: str, token_tracker: TokenTracker, model: str = "gpt-4o-mini") -> None:
    system_prompt = system_prompt_template.format(binary_question)
    for answer in answers_list:
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Answer: {answer.response}"},
                ],
                response_format=BinaryAnswer,
            )
            response = completion.choices[0].message.parsed
            token_tracker.add_usage(
                model,
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                temperture=0.0,
            )
            answer.decision = response.decision
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

# %%
print("\n📊 Purchase Results:")

binarize_answers(survey_data, "Did you purchase any laptop?", token_tracker, evaluator_model)

print(f"Token usage: {token_tracker.get_data()}")
# %%
# # Step 1: Compute purchase rates
purchase_rates = []
for price in prices:
    matching_entries = [e for e in survey_data if e.price == price]
    total = len(matching_entries)
    approved = sum(1 for e in matching_entries if e.decision)
    
    purchase_rate = approved / total if total > 0 else 0
    purchase_rates.append(purchase_rate)

for price, rate in zip(prices, purchase_rates):
    print(f"Price: ${price}, Purchase Rate: {rate:.2%}")

# %%
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Step 2: Linear regression
slope, intercept, r_value, p_value, std_err = linregress(prices, purchase_rates)


print("\n📈 Linear Regression Result:")
print(f"  Slope:       {slope:.4f}")
print(f"  Intercept:   {intercept:.4f}")
print(f"  R-squared:   {r_value**2:.4f}")
print(f"  p-value:     {p_value:.4g}")

if slope < 0 and p_value < 0.05:
    study_result = "✅ Hypothesis supported: Demand trend is significantly downward."
else:
    study_result = "❌ Hypothesis not supported: No significant downward trend."
print(study_result)

# %%
title = f"GPT-Simulated Demand Curve with Linear Regression\nModel: {model}, n_samples: {n_samples}"

# Step 3: Plot
plt.figure(figsize=(8, 5))
plt.plot(prices, purchase_rates, marker='o', label='GPT Purchase Rate')
plt.plot(prices, [slope * p + intercept for p in prices], linestyle='--', color='red', label='Linear Fit')
plt.xlabel("Price ($)")
plt.ylabel("Purchase Probability")
plt.title(title)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    f"{ROOT_DIR}/reports/figures/demand_curve_{model}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Save data
import json

with open(f"{ROOT_DIR}/reports/data/demand_data_{model}.json", "w") as f:
    json.dump({
        "model": model,
        "prices": prices,
        "purchase_rates": purchase_rates,
        "n_responses": len(survey_data),
        "survey_data": [asdict(entry) for entry in survey_data],
        "linear_regression": {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
            "p_value": p_value,
        },
        "study_result": study_result,
        "token_usage": token_tracker.get_data(),
    }, f, indent=4)
print(f"Data saved to {ROOT_DIR}/reports/data/demand_data_{model}.json")

# %%
