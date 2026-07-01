"""
agent/nodes/angle_selector.py

Layer 2 — "The Strategist" — the differentiation gate

Finds the take only JobInGen can credibly own. Must pass the
Differentiation Filter (Spec Section 9) before it's accepted:
  1. Could this exact post appear on any generic career page or
     competitor's feed? If yes -> reject.
  2. Does it speak peer-to-peer, not brand-to-candidate? If no -> rewrite.
  3. Does it tie to something JobInGen can actually do (resume tailoring,
     mock interviews, copilot) when the topic allows?
  4. Does it implicitly answer "why not just use Naukri/LinkedIn?"
  5. Is it free of guarantees, hype, and cringe? If no -> hard fail.
"""

from agent.state import Theme, Angle
from agent.adapters.llm import generate_json


JOBINGEN_CONTEXT = """JobInGen is an AI-powered career platform for
students and early-career job seekers in India. Product surfaces:
- AI resume builder that tailors your resume to each specific job description
- Adaptive AI mock-interview tool with real-time feedback on answers,
  communication, and confidence
- AI career copilot for skill-gap analysis and learning roadmaps
- Curated jobs feed
- Masterclasses and bootcamps with mentors

Anti-positioning — JobInGen is NOT:
- A job board like Naukri (transactional, post-and-pray)
- Generic LinkedIn thought-leadership performing professionalism
- A "guaranteed placement" bootcamp scam
- A motivational guru with no actual tools"""


SYSTEM_INSTRUCTION = f"""You find the take only JobInGen can credibly own
on a trending theme. Reject the generic, could-be-anyone's-career-page
version of this topic.

{JOBINGEN_CONTEXT}

Before answering, check your angle against this filter:
1. Could this exact post appear on any generic career page or
   competitor's feed? If yes, it is NOT differentiated — try again.
2. Does it speak peer-to-peer (a senior who gets it), not
   brand-to-candidate or guru-to-follower?
3. Does it tie to a real JobInGen capability when the topic naturally
   allows it (don't force it if it doesn't fit)?
4. Does it implicitly answer "why not just use Naukri/LinkedIn?"
5. Is it completely free of guarantees, hype, and cringe?

Never promise guaranteed jobs or outcomes.

Output strict JSON only, matching this exact shape:
{{
  "take": "the specific differentiated angle, 1-2 sentences",
  "product_tie": "which JobInGen feature this ties to, or null if none fits naturally",
  "why_only_jobingen": "1-2 sentences on why this couldn't appear on a generic career page",
  "passes_filter": true
}}
If you cannot find an angle that passes all 5 filter checks, set
"passes_filter": false and explain why in "why_only_jobingen"."""


def run_angle_selector(
        theme: Theme,
        tension: str,
        feedback: str | None = None,
    ) -> Angle | None:
    """
    Returns an Angle if it passes the differentiation filter, or None
    if nothing credible was found (a valid outcome — better to skip a
    post than ship a generic one).
    """
    user_prompt = (
        f"Theme: {theme.label}\n"
        f"Audience tension: {tension}\n\n"
        "Find the differentiated JobInGen angle on this."
    )
 
    if feedback:
        user_prompt += (
            f"\n\nIMPORTANT: a previous attempt on this same theme failed "
            f"brand+safety review. Specific feedback to address: {feedback}\n"
            f"Produce a genuinely different angle that fixes this issue -- "
            f"do not just reword the same idea."
        )

    result = generate_json(
        system_instruction=SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
    )

    if not result.get("passes_filter", False):
        print(f"[angle_selector] No angle passed the differentiation filter. "
              f"Reason: {result.get('why_only_jobingen', 'unspecified')}")
        return None

    angle = Angle(
        take=result.get("take", ""),
        product_tie=result.get("product_tie"),
        why_only_jobingen=result.get("why_only_jobingen", ""),
    )

    print(f"[angle_selector] Angle: {angle.take}")
    print(f"[angle_selector] Product tie: {angle.product_tie}")
    return angle


# Quick manual test — run this file directly
if __name__ == "__main__":
    from agent.nodes.trend_scout import run_trend_scout
    from agent.nodes.signal_synthesizer import run_signal_synthesizer
    from agent.nodes.trend_scorer import run_trend_scorer
    from agent.nodes.audience_modeler import run_audience_modeler

    signal = run_trend_scout()
    themes = run_signal_synthesizer(signal)
    chosen, _ = run_trend_scorer(themes)

    if chosen:
        tension = run_audience_modeler(chosen)
        angle = run_angle_selector(chosen, tension)
        if angle:
            print(f"\nWhy only JobInGen: {angle.why_only_jobingen}")
        else:
            print("\nNo differentiated angle found this run.")
    else:
        print("\nNo strong theme this run.")