# How the FT tells stories with data

Date: 2026-10-30

**Question.** What concrete, citable rules does the Financial Times follow when it tells stories with data, and which of them can a design-review persona apply to Lookout, a personal weather dashboard (current conditions, rain vs normal, water-year cumulative rain, rain events, temperature context, solar energy) viewed mostly on a phone?

**Method and limits.** ft.com blocks automated fetches, so none of the Chart Doctor columns and none of *How Charts Work* were read directly. The rules below come from (a) the FT's own Visual Vocabulary in the `Financial-Times/chart-doctor` repo and (b) first-party interviews and talks with Alan Smith and John Burn-Murdoch. Where a quote reaches us through a third-party write-up of a talk, the rule says so. Anything not traced to a source is under [Unverified](#unverified), not in Rules.

## Rules

Source tags: **VV** = FT Visual Vocabulary README; **JBM-iMEdD** = write-up of a Burn-Murdoch talk; **JBM-Eye** = Eye magazine interview; **AS** = Alan Smith interviews.

### Choosing the chart

1. **Choose the chart by the relationship in the data, not by habit.** The FT groups chart types into nine relationships: Deviation, Correlation, Ranking, Distribution, Change over Time, Part-to-whole, Magnitude, Spatial, Flow. Smith's test is whether you are "picking out and emphasising key statistical relationships in your data in a way that will make sense to users" ([VV](https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/README.md); [AS, Visualising Data](https://visualisingdata.com/2016/04/six-questions-alan-smith/)).
2. **Deviation charts emphasise variation from a fixed reference point, and the reference can be zero, a target or a long-term average.** The listed forms are diverging bar and surplus/deficit filled line, the latter showing a balance "either against a baseline or between two series" ([VV](https://raw.githubusercontent.com/Financial-Times/chart-doctor/main/visual-vocabulary/README.md)).
3. **Change-over-time charts "give emphasis to changing trends", and "choosing the correct time period is important to provide suitable context for the reader."** The line is "the standard way to show a changing time series" ([VV](https://raw.githubusercontent.com/Financial-Times/chart-doctor/main/visual-vocabulary/README.md)).
4. **Use area charts with care.** The FT says they are good at showing changes to a total, but "seeing change in components can be very difficult" ([VV](https://raw.githubusercontent.com/Financial-Times/chart-doctor/main/visual-vocabulary/README.md)).
5. **A calendar heatmap shows temporal patterns "at the expense of showing precision in quantity."** Use it when the pattern matters more than the exact value ([VV](https://raw.githubusercontent.com/Financial-Times/chart-doctor/main/visual-vocabulary/README.md)).
6. **Correlation is read as causation unless you say otherwise.** The Vocabulary warns that readers often assume a relationship is causal ([VV](https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/README.md)).

### Axis honesty

7. **Magnitude columns must start at zero.** The Vocabulary says the column "must always start at 0 on the axis", and that magnitude charts usually show counted numbers, not calculated rates or percentages ([VV](https://raw.githubusercontent.com/Financial-Times/chart-doctor/main/visual-vocabulary/README.md)). The one stated exception is the lollipop, which "does not HAVE to start at zero (but preferable)" (same source).
8. **Be explicit about scale choices; the FT uses log scales when growth is exponential.** Smith says the FT is willing to run "log scales to understand exponential growth", even offering a reader toggle between linear and log ([AS, Stats + Stories](https://statsandstories.net/economics1/start-with-the-charts)). Rainfall is not exponential, so this rule mostly tells a reviewer to ask why a non-linear axis is there.

### Telling the story on the chart

9. **Write a title that makes a point.** Burn-Murdoch: "if you give it a title that makes a point, you're going to get a stronger reaction," versus a neutral label like "Number of Covid cases by day" ([JBM-Eye](https://eyemagazine.com/feature/article/conversation-with-john-burn-murdoch)). He says he puts "a lot of effort and time into getting that title right" ([JBM-iMEdD](https://lab.imedd.org/en/from-data-to-storytelling-concept-and-design-tips-from-the-financial-times-john-burn-murdoch/)). Smith writes chart titles first, then "write[s] around them" ([AS, Stats + Stories](https://statsandstories.net/economics1/start-with-the-charts)).
10. **Annotate on the chart.** "We do think hard about annotations and the use of text in our charts," and "the text is critical" ([JBM-Eye](https://eyemagazine.com/feature/article/conversation-with-john-burn-murdoch)). "Text is where people's attention goes first. If we're not using that, we're really missing an opportunity" ([JBM-iMEdD](https://lab.imedd.org/en/from-data-to-storytelling-concept-and-design-tips-from-the-financial-times-john-burn-murdoch/)).
11. **Label series directly instead of using a legend.** Burn-Murdoch: "You're asking someone to keep checking back and forth between the legend and the lines." He moved labels onto the lines ([JBM-iMEdD](https://lab.imedd.org/en/from-data-to-storytelling-concept-and-design-tips-from-the-financial-times-john-burn-murdoch/); quote verbatim, example is the write-up's).
12. **Highlight one thing and mute the rest.** The write-up gives his maxim as "Minimize distraction. Maximize contrast", with key data in a striking colour and secondary data in grey ([JBM-iMEdD](https://lab.imedd.org/en/from-data-to-storytelling-concept-and-design-tips-from-the-financial-times-john-burn-murdoch/)). The grey-context and single-red-line detail is the writer's paraphrase, not a quote.
13. **Show what is normal first, then what is different.** Describing how to present 2020-21 against earlier years, he says to "draw a few lines to show someone, 'This is normal,'" then "bring in the thing that is different" ([JBM-Eye](https://eyemagazine.com/feature/article/conversation-with-john-burn-murdoch)).
14. **Include no more information than the chart needs.** "We try to ensure that our charts contain no more information than they need to," and the team is "opposed to 'chartjunk'" ([JBM-Eye](https://eyemagazine.com/feature/article/conversation-with-john-burn-murdoch)). He also says a product is complete "not when nothing more can be added, but when nothing more can be taken away from it" ([JBM, Visualising Data](https://visualisingdata.com/2015/09/six-questions-with-john-burn-murdoch/)).
15. **Fewer, better-focused charts that carry part of the narrative.** Smith: "We can use these graphics to help carry some of the narrative so that it's not duplicative," and "we will be making fewer charts, but they will be better focused" ([AS via Digiday](https://digiday.com/uk/financial-times-guide-data-visualization/)).

### Phones

16. **Design mobile-first.** Smith: "we do need to make sure that we're sort of serving mobile first content" ([AS, Stats + Stories](https://statsandstories.net/economics1/start-with-the-charts)). Burn-Murdoch says he started too late and that 40 percent of FT readers arrive in a way that makes non-responsive interactive graphics "completely invisible" ([JBM, Visualising Data](https://visualisingdata.com/2015/09/six-questions-with-john-burn-murdoch/)).
17. **Prefer self-explaining annotated graphics to interactivity.** Martin Stabe (FT head of interactive news, 2016): "interactivity is quite hard for people to use, so we want to use it sparingly" ([Journalism.co.uk](https://www.journalism.co.uk/watch-how-the-data-team-at-the-ft-is-moving-forward-as-audience-go-mobile/)). Smith: "static, well designed, static graphics are still the lifeblood of data visualization" ([AS, Stats + Stories](https://statsandstories.net/economics1/start-with-the-charts)).
18. **Build each chart to render at every size, not shrink one.** D3 output as SVG yields "a mobile phone-sized version, a desktop-sized version, a print version, a social media version and a video version" from one chart ([JBM-Eye](https://eyemagazine.com/feature/article/conversation-with-john-burn-murdoch)). The source describes the mechanism (per-format versions), not a stated "redraw, don't shrink" rule; that wording is the reviewer's reading.

## Applying to Lookout

The mapping from FT rule to Lookout chart is my inference, not FT guidance. Rule numbers refer to the list above.

| Lookout view | FT relationship | Rules to check |
|---|---|---|
| Rain vs normal (this month or year against climatology) | Deviation, with the long-term average as reference (R2) | Is the reference line the "normal"? Show normal first, then the current period (R13). Title states the finding, such as "March ran 40% wetter than normal" (R9). Single highlight colour for the current period, grey for normals (R12). If bars are used, zero baseline (R7). |
| Water-year cumulative rain | Change over time (R3) | Line, not area (R4). Window is the water year, so the time period gives context (R3). Overlay prior years or normal in grey and the current year highlighted (R12). Label lines directly at the right edge (R11). Annotate the wettest storms or the gap to normal (R10). |
| Rain events | Magnitude of counted events (R7) or timeline of events | Bars start at zero (R7). Calendar heatmap only if pattern beats precision (R5). Annotate the biggest event (R10). |
| Temperature context (today vs normal range) | Deviation or distribution against normal (R2) | Show the normal range in grey and today highlighted (R12). Plain-language title (R9). |
| Solar energy | Change over time, or magnitude (R3, R7) | Daily totals as zero-based columns (R7). Compare to clear-sky or typical output only with a labelled reference (R10). Do not imply cloud caused a dip without saying so (R6). |
| Current conditions | Not a chart-vocabulary case | Apply R14: no more than the glance needs. |
| Whole dashboard on a phone | n/a | Mobile-first (R16). Prefer static annotated charts to hover and tooltip interactions (R17). Check each chart at phone width as its own layout (R18). Few charts, each doing a job (R15). |

## Sources

- FT Visual Vocabulary README: <https://github.com/Financial-Times/chart-doctor/blob/main/visual-vocabulary/README.md> (also raw: <https://raw.githubusercontent.com/Financial-Times/chart-doctor/main/visual-vocabulary/README.md>). Repo: <https://github.com/Financial-Times/chart-doctor>. The folder also holds `Visual-vocabulary-en.pdf` and `poster.png`, which I did not read.
- Eye magazine, conversation with John Burn-Murdoch: <https://eyemagazine.com/feature/article/conversation-with-john-burn-murdoch>
- Visualising Data, Six questions with Alan Smith (2016): <https://visualisingdata.com/2016/04/six-questions-alan-smith/>
- Visualising Data, Six questions with John Burn-Murdoch (2015): <https://visualisingdata.com/2015/09/six-questions-with-john-burn-murdoch/>
- Stats + Stories, "Start With the Charts" with Alan Smith: <https://statsandstories.net/economics1/start-with-the-charts>
- Digiday, "The Financial Times guide to data visualization": <https://digiday.com/uk/financial-times-guide-data-visualization/>
- Journalism.co.uk, FT data team and mobile (Martin Stabe, 2016): <https://www.journalism.co.uk/watch-how-the-data-team-at-the-ft-is-moving-forward-as-audience-go-mobile/>
- iMEdD Lab write-up of a Burn-Murdoch talk (third-party report of his words): <https://lab.imedd.org/en/from-data-to-storytelling-concept-and-design-tips-from-the-financial-times-john-burn-murdoch/>

## Unverified

Could not be traced to a primary source. Do not cite these as FT rules.

- **Standfirst/subtitle and source line.** No source reached says what the FT requires for either. Treat any such rule as house style the reviewer invents, not FT guidance.
- **Truncated line-chart axes and dual axes.** No FT source reached states a rule. Statements found came from non-FT sites (Handsondataviz, Pomegra, Nastengraph). The only FT axis rule traced is the zero baseline for columns (R7).
- **Missing data.** No FT source reached addresses it. The Vocabulary lists Uncertainty as an unfilled "Todo".
- **Chart Doctor columns (ft.com).** Not read; ft.com is blocked to automated fetches. No column was quoted. Titles of specific columns on annotation, log scales or mobile remain unchecked.
- **Alan Smith, *How Charts Work* (2022).** No first-party excerpt reached. One reader's blog summary said the book advises titles be "engaging, simple, accurate", but that is secondhand and excluded.
- **FT Origami / o-colors and chart colour palette.** `github.com/Financial-Times/o-colors` returned 404 and no Origami data-vis documentation was found. FT colour rules are unverified.
- **"Redraw, not shrink" as an FT rule.** A search result summarising a Cambridge *Data Journalism Handbook* chapter says the FT rotated a parliament seat chart 90 degrees for mobile. I could not open the page (403). Treat as a lead only.
- **Grey-context highlighting and "legend vs labels" quotes** come via the iMEdD write-up of a talk, not the talk itself. The red-line-among-grey detail is the writer's paraphrase.
- **NPR "Indicator" interview with Burn-Murdoch.** Search snippet said he writes sentence titles; the transcript fetch failed, so not used.
- **Alan Smith Dual Axis Podcast.** Page unreachable.
