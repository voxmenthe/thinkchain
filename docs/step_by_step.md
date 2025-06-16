We would like to implement a parallel version of this implementation (key features of the current codebase are highlighted in `docs/codebase_overview.md` and you should read this as well.).
We will leave the existing implementation in place and create new files. 

Follow the instructions in @P2_web_research.md  for this goal.

The new implementation will use the new google-genai SDK instead of anthropic (documentation can be found at https://googleapis.github.io/python-genai/ and also here https://ai.google.dev/gemini-api/docs/text-generation).

Note that the SDK and its documentation has recently changed, so it is **very important to read the current documentation** to understand how to use it, and make sure we are using the latest version - you **CANNOT** use your existing knowledge here but must instead make sure to read the documentation, understand it, and document your findings, and use those in your implementation.

------------------------

We would like to implement a parallel version of this implementation (key features of the current codebase are highlighted in `docs/codebase_overview.md`).
We will leave the existing implementation in place and create new files. Follow the instructions in @P2_web_research.md for this goal.

The new implementation will use the new google-genai SDK instead of anthropic (documentation can be found at https://googleapis.github.io/python-genai/ and also here https://ai.google.dev/gemini-api/docs/text-generation).

Note that the SDK and its documentation has recently changed, so it is **very important to read the current documentation** to understand how to use it, and make sure we are using the latest version - you **CANNOT** use your existing knowledge here but must instead make sure to read the documentation, understand it, and document your findings, and use those in your implementation.

------------------------

Ok now go ahead with the instructions in @P3_implementation_plan.md - make sure to include most of the key parts of the code as snippets with instructions suitable for immediate implementation by a junior engineer. Again, make doubly sure to re-check the documentation and make sure you are using the latest version of the SDK (documentation can be found at https://googleapis.github.io/python-genai/ and also here https://ai.google.dev/gemini-api/docs/text-generation).


+++++++++++++++++++++++++

We would like to update this implementation to allow for the use of different LLM providers - in particular we want to add gemini from the google-genai SDK, in addition to Anthropic. Our task right now is to improve and substantially expand the implmentation plan in `implementation_plan.md` and create a new version that improves the planned implementation and includes any/all necessary changes to the codebase to support this.

Key features of the current codebase are highlighted in `codebase_overview.md` and there is additional motivation in `blog.md`. In particular the async tool call that allows for thinking between tool calls is a key feature that we want to make sure works across LLM providers.

The new implementation will use the new google-genai SDK instead of anthropic (documentation can be found at https://googleapis.github.io/python-genai/ and also here https://ai.google.dev/gemini-api/docs/text-generation).

Note that the SDK and its documentation has recently changed, so it is important to read the current documentation to understand how to use it, and make sure we are using the latest version - you **CANNOT** use your existing knowledge here but must instead make sure to read the documentation, understand it, and document your findings, and use those in your implementation.

