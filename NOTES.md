## SUBMISSION NOTES

# Problem Identification
- Major problem was the lack of structured JSON output - the raw text does not satisfy the fields that need to be populated
- No validation for the structuring - there is only a defined set of accepted categories and the receipt amount has to be a float
- Other major problem - error handling
  - errors would propagate with initial scaffold if the model output was malformed, didn't return anything, or was unreachable
  - not knowing the source of errors makes it impossible to validate and debug agentic system

# What I Built and Left Out
- implemented LineItems and Category classes for Pydantic model output structuring and validation
- introduced system prompt to clarify structuring task for LLM
- brought model calling outside endpoint to easily configure model parameters and create mock tests without hitting model
- created error handling for unreachable model API and malformed/empty output
- implemented retry logic for malformed or empty model responses (not model unreachable) and limited after max amount of retries
- structured error JSON response with error type and details
- distinguished model output unusable from model unreachable errors
- Logged model outputs and failures to md file
- Implemented tests in phases of implentation (wrote tests for JSON structuring first (with and without hitting model)), error handling, and corrective retries
- Added simple checks to ensure that model output is validated against excessive length or simple injection attacks or unwanted characters
- Left out more complex logic for resolving to category - LLM can handle simply
- Left out testing and resolving more complex input strings for receipts - or other types of data handling (images, etc)
- Left out validating data accuracy - more ambiguous problem to validate against input string
- Did not implement a frontend dashboard or UI for the user to enter receipt text - can be done but would take up too much time and not core functionality

# Break Points and Fixes with More Time
- Model hallucinations - validation is for structure not model accuracy. This is trusting that the LLM provider does not hallucinate with this straightforward tasks. Would implement more deterministic checking of model output vs input string to ensure that the data is accurate.
- Logging is pretty rudimental - does not have robust handling for multiple concurrent reads/writes way and no compaction/truncaction of logs can balloon size and storage.
- Handling unexpected exceptions in an outer layer
- No retry if the API is unreachable - could implement a router between different model providers if one does not work
- Would implement intermediate logging for retries as well
- Log metrics for agent performance (receipts parsed, total input/output tokens)
- Didn't evaluate different models (just stuck with model provided)
- Caching for recent/similar requests to minimize API calls


# AI Tool Usage
- I used Claude Code (Opus 4.8, medium) to initially analyze the scaffold file and potential errors around output validity.
- Orchestrated implementation/testing in phases. JSON structuring, error handling, retry, and logging and more validation, all with tests written and asserted at each stage.
- Pushed back on some early architectural decisions, gave clear instructions for initial refactor (creating seperate call model function, system prompt, class for categories, not writing all tests at once, using TestClient and mock hits for tests)
- Used review my decisions and push back with tradeoffs and limitations
- Generated manual test cases to test server and model hits in another terminal