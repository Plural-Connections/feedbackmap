#!/usr/bin/env python3

"""
Generates fake survey data for people in different states
by asking GPT-3 the prompt below.   
"""

import csv
import openai
import sys

_PROMPT = """
Please generate more data in this format:

- State: Massachusetts
- Age: 23
- Are you left or right handed: Left
- Favorite ice cream flavor: Mint chip
- Favorite sport: Basketball
- What is your favorite meal: Broiled lobster with butter garlic herb sauce
- What is a funny thing you saw on the street recently: A snowplow with a "Baby on board" sign
- What is the weather like where you live: It's wicked cold here now, in the teens.

- State:
"""

_STATES = {
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}


def run_completion_query(prompt, model="text-davinci-003", num_to_generate=1):
    tries = 0
    while tries < 3:
        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=1.0,
                n=num_to_generate,
                stop=["\n\n"],
                max_tokens=300,
            )
            return response
        except (openai.error.RateLimitError, openai.error.APIError) as e:
            tries += 1
            time.sleep(10)


if __name__ == "__main__":

    prompt_questions = set(
        [
            x.split(":")[0].replace("- ", "")
            for x in _PROMPT.split("\n")
            if x.startswith("-")
        ]
    )
    csv_writer = csv.DictWriter(sys.stdout, prompt_questions)
    csv_writer.writeheader()

    for i in range(20):  # Generate 20 responses for each state
        for state in _STATES.values():
            prompt = _PROMPT.strip() + " " + state
            res = run_completion_query(prompt)
            x = {q: "" for q in prompt_questions}
            x["State"] = state
            text = res["choices"][0]["text"]
            lines = text.split("\n")
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    question, answer = parts
                    question = question.replace("- ", "").strip()
                    if question in prompt_questions:
                        answer = answer.strip()
                        x[question] = answer

            # In practice there'll rarely be an exception here, but we need
            # to guard against it happening since there's no guarantee
            try:
                age = int(x["Age"])
                if age < 20:
                    age = "Younger than 20"
                else:
                    age = "%d-%d" % (10 * (age // 10), 10 * (age // 10) + 9)
                x["Age"] = age
            except (ValueError, KeyError):
                x["Age"] = ""  # Leave it blank if it does happen

            csv_writer.writerow(x)
