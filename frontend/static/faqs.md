1. **I'd like to try Feedback Map out but don't have any survey data.  What should I do?**

You can try it out with our [fake survey data](https://github.com/Plural-Connections/feedbackmap/blob/main/test_data/fake_survey.csv)!  The fake data includes responses generated from GPT-3 to questions like "What is your favorite meal?" and "What is a funny thing you saw on the street recently?"

2. **Does Feedback Map support analysis of non-English text?**

At the moment, the language models powering Feedback Map are most accurate on/relevant for English texts only.  We hope to add in support for other languages as we grow our team.    

3. **Can I import outputs from other survey platforms beyond Google Forms?**

You can, but there's no guarantee it will work :)  For now, we've designed Feedback Map to primarily support analysis of Google Survey outputs, but also hope to support other formats in the future.

4.  **How much should I trust the outputs?**

We believe the scatterplots might be easier to evaluate the validity of (since users are able to explore individual comments provided by survey respondents directly).  The auto-generated summaries are produced using Open AI's GPT-3 model; it is less transparent/obvious how this model processes the underlying survey feedback to produce these summaries.  We encourage Feedback Map users to take all summaries and analyses presented through the platform with a grain of salt, and to use other methods (community validation, prior knowledge, etc.) to determine how trustworthy the presented patterns are.  We plan to continue exploring ways of validating the outputs of the language models powering Feedback Map in order to better support such determinations moving forward.