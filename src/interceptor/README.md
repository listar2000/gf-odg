### LLM Interceptor Design Doc

This component is meant to be an optional plug-in (or in the language of software development, a middleware) to be used in the generation process of a pretrained language model. 

Specifically, in the simple case, we are given a set of "trigger words" {w1, w2, ..., wn} to be detected. Note that each trigger word can be either one or more tokens. During generation, suppose the LLM has generated a stream of tokens [x1, x2, ..., xt] so far at time t, then we need to check if any of the "last few tokens" [x(t-k), ..., xt] matches any of the trigger words. If yes, we will take some action to the stream, which may include stopping generation, editing, etc. (but the specific action can be user-defined). 

(Note that in reality, the generation would be in a batch, so we also want to support efficient detection for all sequences simultaneously, perhaps using some torch tensor operations).

In more complicated cases, a predefined set of trigger words might not be enough, so we might even need to leverage another language model to detect the generated text in real time. What this means is that even the inteception algorithm might have multiple implementations. For this reason, this module is organized as:

- `src.interceptor.base_interceptor`: The base class for all interceptors, defining the interface
- `src.interceptor.trigger_interceptor`: A simple interceptor that checks if any of the trigger words are present in the last few tokens
- `src.interceptor.llm_interceptor`: A more complex interceptor that leverages another language model to detect the generated text in real time