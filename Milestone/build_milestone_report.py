#!/usr/bin/env python3
"""Build the A4 milestone draft from the archived development results."""
import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Table, TableStyle,
)

MILESTONE = Path(__file__).resolve().parent
REPOSITORY = 'https://github.com/wzhnbsixsixsix/CS5491-RA-NSC'
MILESTONE_URL = f'{REPOSITORY}/tree/milestone'
METHODS = {
    'original_nsc': 'NSC',
    'position_example': 'Position',
    'student': 'RA',
    'ablation': 'No skip',
    'reported_parameters': 'Params',
    'reference_flops': 'FLOPs',
    'random': 'Random',
}


def register_fonts(directory):
    """Embed the requested font family and its emphasized faces."""
    files = {'TNR': '', 'TNR-Bold': ' Bold', 'TNR-Italic': ' Italic'}
    for name, suffix in files.items():
        pdfmetrics.registerFont(TTFont(name, str(directory / f'Times New Roman{suffix}.ttf')))
    pdfmetrics.registerFontFamily('TNR', normal='TNR', bold='TNR-Bold', italic='TNR-Italic')


def build_report(output, development, mechanism):
    """Lay out the method, numerical checks, and both preliminary result tables."""
    margin = 2.5 * cm
    width, height = A4
    body_width = width - 2 * margin
    title = 'Residual-Aware Neural Spectral Capacity for Transformer Architecture Selection'
    report = BaseDocTemplate(
        str(output), pagesize=A4, title=title, author='Wang Zenghuan; Zhao Deyi', invariant=1,
    )
    frame = Frame(margin, margin + 15, body_width, height - 2 * margin - 15,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    def number_page(canvas, document):
        """Number each A4 page in the report font."""
        canvas.setFont('TNR', 12)
        canvas.drawCentredString(width / 2, margin, str(document.page))

    report.addPageTemplates(PageTemplate(frames=[frame], onPage=number_page))
    body = ParagraphStyle('Body', fontName='TNR', fontSize=12, leading=14, spaceAfter=7)
    heading = ParagraphStyle('Heading', parent=body, fontName='TNR-Bold',
                             spaceBefore=7, spaceAfter=4, keepWithNext=True)
    center = ParagraphStyle('Center', parent=body, alignment=1)
    centered_title = ParagraphStyle('Title', parent=center, fontName='TNR-Bold')
    cell = ParagraphStyle('Cell', parent=body, spaceAfter=0)
    code_guide = ParagraphStyle('CodeGuide', parent=body, spaceAfter=4)
    story = []

    def paragraph(content, style=body):
        """Append a paragraph with the selected report style."""
        story.append(Paragraph(content, style))

    def milestone_file(path, label=None):
        """Link to a file in the exact code and results snapshot."""
        return f'<link href="{REPOSITORY}/blob/milestone/{path}"><u>{label or path}</u></link>'

    def table(rows, widths):
        """Append a result table with a repeating header."""
        content = [[Paragraph(str(value), cell) for value in row] for row in rows]
        result = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT', spaceAfter=7)
        result.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8EEF1')),
            ('LINEABOVE', (0, 0), (-1, 0), 0.6, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), 0.4, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 0.6, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(result)

    paragraph(title, centered_title)
    paragraph('CS5491 Artificial Intelligence | Project milestone report draft<br/>'
              'Wang Zenghuan and Zhao Deyi | 20 November 2026', center)
    paragraph('1. Problem and motivation', heading)
    paragraph('Selecting a Transformer architecture can require training many candidates. Neural '
              'Spectral Capacity (NSC) gives a training-free score from matrix dimensions and an '
              'initialization convention [1]. Its additive FFN term scores the two projection '
              'matrices separately. The FFN sub-block instead combines a branch with an identity '
              'shortcut. We ask whether scoring this addition improves the ranking of trained '
              'architectures and the choice of candidates under a total parameter cap.')
    paragraph('2. Proposed score and mechanism ablation', heading)
    paragraph('Let <i>d</i> be the hidden width and <i>f</i> the FFN width. In a local linear '
              'reference model, the branch Jacobian is <i>B</i> = <i>W</i><sub>2</sub> <i>D</i> '
              '<i>W</i><sub>1</sub> and the residual Jacobian is <i>I</i> + <i>B</i>. We model the '
              'two projections with independent zero-mean Gaussian entries (standard deviation '
              '0.02), the diagonal gate entries with independent Bernoulli(0.5) draws, and '
              'normalization with the identity map. These assumptions define the reference '
              'calculation.')
    paragraph('Write <i>C</i>(<i>A</i>) = ln det(<i>I</i> + <i>A</i><sup>T</sup><i>A</i>). '
              'Our residual-aware score (RA) replaces the FFN term by '
              'E[<i>C</i>(<i>I</i> + <i>B</i>)] − <i>d</i> ln 2. The subtraction makes a '
              'zero branch contribute zero. The shortcut ablation uses E[<i>C</i>(<i>B</i>)]. '
              'Both methods retain the starter’s per-head attention term and sum across layers.')
    paragraph('Each estimate averages 32 branch draws per FFN shape with a fixed seed (202627) '
              'keyed by (<i>d</i>, <i>f</i>). Conditional on the number of active gates, sampled '
              'weights at the active indices have the same distribution as the original gated '
              'projections. We average the residual contributions for <i>B</i> and −<i>B</i> '
              'to cancel odd terms in the branch perturbation. They share the same distribution '
              'and the same ablation capacity. Cholesky factors give the log determinants. '
              'The program caches both means per shape and scores from the configuration alone.')
    paragraph('3. Benchmark and protocol', heading)
    paragraph('The course archive attributes 200 GPT-2 / Transformer-XL configurations to '
              'LiteTransformerSearch [2]. This milestone uses the 100 development architectures '
              'and their validation perplexities (PPL). Another 100 architectures from the '
              'same archive are reserved for final assessment after freezing the method. The '
              'benchmark includes variation in width, depth, heads, and FFN widths.')
    paragraph('We compare RA and its ablation with original NSC, the supplied position heuristic, '
              'total parameters, reference FLOPs, and uniform random selection. Ranking uses '
              'negative PPL because higher architecture scores predict lower PPL. Budget selection '
              'uses the same feasible pool and number of picks for every method.')
    paragraph('Code and reproducibility', heading)
    paragraph(f'<link href="{MILESTONE_URL}"><u>{MILESTONE_URL}</u></link>', code_guide)
    paragraph(
        '<b>Start here.</b> '
        f'{milestone_file("nsc-starter/student_score.py")} implements the RA score and shortcut '
        f'ablation. {milestone_file("nsc-starter/evaluate.py")} runs the ranking and budgeted '
        f'selection. {milestone_file("Milestone/check_residual.py")} checks the mechanism and '
        'sampling error.',
        code_guide,
    )
    paragraph(
        '<b>Reproduce.</b> '
        f'{milestone_file("Milestone/README.md", "Run commands")}, '
        f'{milestone_file("nsc-starter/data/provenance.json", "dataset provenance")}, '
        f'and {milestone_file("Milestone/build_milestone_report.py", "report builder")}.',
        code_guide,
    )
    paragraph(
        '<b>Archived results.</b> '
        f'{milestone_file("Milestone/results/milestone_development.json", "Development ranking and selection")} '
        f'and {milestone_file("Milestone/results/milestone_mechanism.json", "mechanism and sampling checks")}.',
        code_guide,
    )
    story.append(PageBreak())

    paragraph('4. Mechanism and numerical checks', heading)
    paragraph('The implementation returned zero for a zero branch, matched a singular-value '
              'calculation for a fixed small matrix, preserved layer-reorder invariance, and '
              'agreed on equivalent scalar and per-layer configurations. With weak branches '
              '<i>B</i> = 0.001<i>I</i> and <i>B</i> = 0.001[[0, 1], [−1, 0]], the ablation '
              'gave the same capacity (about 2.00 × 10<sup>−6</sup>). The sign-averaged residual '
              'contributions were about 2.50 × 10<sup>−13</sup> and 1.00 × 10<sup>−6</sup>. '
              'The difference isolates the effect of how the branch combines with the identity.')
    paragraph('<b>Table 1. Relative differences from an independent higher-draw estimate.</b>')
    rows = [['<b>(d, f)</b>', '<b>Reference draws</b>', '<b>RA</b>', '<b>No skip</b>']]
    for row in mechanism['comparisons']:
        scores = row['scores']
        rows.append([f"({row['hidden_width']}, {row['ffn_width']})", row['reference_draws'],
                     f"{scores['residual']['relative_error']:.2%}",
                     f"{scores['ablation']['relative_error']:.2%}"])
    table(rows, [107, 127, 108, body_width - 342])
    paragraph('The 8 × 16 case uses explicit Bernoulli masks and SVD. Its larger relative '
              'differences occur on very small capacity values and fall within two combined '
              'Monte Carlo standard errors. The other rows use independent 256-draw references. '
              'These checks describe estimator precision at sampled shapes.')
    paragraph('5. Preliminary development ranking', heading)
    paragraph('<b>Table 2. Ranking against negative validation PPL.</b> Spearman uses all 100 '
              'architectures. Kendall τ-b uses 4,950 pairs overall and 180 size-matched pairs '
              'with equal hidden width and total-parameter ratio min/max ≥ 0.90.')
    rows = [['<b>Scoring method</b>', '<b>Spearman ρ</b>', '<b>All τ-b</b>',
             '<b>Matched τ-b</b>', '<b>Score ties¹</b>']]
    for key, result in development['ranking'].items():
        all_pairs = result['all_pairs']
        rows.append([METHODS[key], f"{result['spearman_rho']:.4f}",
                     f"{all_pairs['tau_b']:.4f}", f"{result['size_controlled_pairs']['tau_b']:.4f}",
                     all_pairs['score_only_ties'] + all_pairs['both_ties']])
    table(rows, [110, 93, 72, 94, body_width - 369])
    paragraph('¹ Number of all-panel architecture pairs tied on a method’s score. The matched '
              'pairs can share architectures. Timings and detailed tie counts are archived in the '
              'development JSON.')
    ranking = development['ranking']
    ra, ablation, nsc = (ranking[key] for key in ('student', 'ablation', 'original_nsc'))
    paragraph(f"RA’s overall τ-b is {ra['all_pairs']['tau_b']:.4f}, versus "
              f"{ablation['all_pairs']['tau_b']:.4f} for the otherwise identical shortcut "
              f"ablation and {nsc['all_pairs']['tau_b']:.4f} for original NSC. Matched τ-b is "
              f"{ra['size_controlled_pairs']['tau_b']:.4f} for RA and "
              f"{ablation['size_controlled_pairs']['tau_b']:.4f} for the ablation. This comparison "
              "uses overlapping pairs from the development split and is exploratory.")
    story.append(PageBreak())

    paragraph('6. Preliminary budgeted selection', heading)
    paragraph('<b>Table 3. Mean best selected PPL / regret.</b> Lower is better for both. '
              'Regret subtracts the pool’s minimum PPL (“Best”). Each entry averages 200 fixed-seed '
              'trials that shuffle ties or draw random candidates without replacement. '
              'The feasible pool has <i>n</i> architectures. Displayed means are rounded independently.')
    rows = [['<b>Cap</b>', '<b><i>k</i></b>', '<b><i>n</i></b>', '<b>Best</b>']
            + [f'<b>{label}</b>' for label in METHODS.values()]]
    for row in development['selection']:
        values = [f"{row['parameter_cap'] // 1_000_000}M", row['k'], row['eligible'],
                  f"{row['oracle_ppl']:.2f}"]
        for key in METHODS:
            output_stats = row['methods'][key]
            values.append(f"{output_stats['mean_best_ppl']:.2f}<br/>{output_stats['mean_regret']:.2f}")
        rows.append(values)
    table(rows, [33, 17, 24, 43, 43, 49, 37, 50, 48, 45, body_width - 389])
    paragraph('7. Interpretation and remaining work', heading)
    paragraph('Both FFN branch scores improve overall ranking relative to NSC. RA improves '
              'matched τ-b by about 0.011 over the ablation, and their regrets match in all '
              'eight settings. Against original NSC, RA reduces regret at 10M with one pick '
              'and 20M with three picks. At 50M with one pick, regret rises from 0.09 to 1.55. '
              'The development evidence for an incremental shortcut benefit is limited to matched ranking.')
    paragraph('The archive has one recorded trained endpoint per architecture and omits training '
              'seeds and parts of the training protocol. These results describe a fixed pool and '
              'a simplified local block model. Next we will check sensitivity to the estimator '
              'settings, freeze the scoring and evaluation choices, and evaluate the reserved '
              'architectures. The reported random-trial variation measures selection uncertainty.')
    seconds = development['scoring_seconds']
    paragraph(f'The first RA scoring pass, including both FFN means, took {seconds["student"]:.1f} s on '
              'an Apple M1 MacBook Pro with 16 GB RAM (Python 3.12.14, NumPy 2.3.5, Accelerate). '
              f'Original NSC took {seconds["original_nsc"]:.2f} s. The ablation reuses the FFN means. '
              f'The {milestone_file("Milestone/README.md", "milestone README")} gives the exact '
              'commands and paths to the JSON outputs.')
    paragraph('References', heading)
    paragraph('[1] C. Zhu, R. Zhao, and Z. Lu. <i>Neural Spectral Capacity</i>. '
              '<link href="https://arxiv.org/abs/2609.23087"><u>arXiv:2609.23087</u></link>, 2026.')
    paragraph('[2] M. Javaheripi et al. <i>LiteTransformerSearch</i>. NeurIPS, 2022. '
              '<link href="https://arxiv.org/abs/2203.02094"><u>arXiv:2203.02094</u></link>.')
    report.build(story)


def main():
    """Read the archived inputs and write the submission PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-dir', type=Path, required=True,
                        help='Directory containing Times New Roman regular, bold and italic TTFs')
    parser.add_argument('--output', type=Path, default=MILESTONE / 'milestone_report.pdf')
    args = parser.parse_args()
    register_fonts(args.font_dir)
    development = json.loads((MILESTONE / 'results/milestone_development.json').read_text())
    mechanism = json.loads((MILESTONE / 'results/milestone_mechanism.json').read_text())
    if development['split'] != 'development' or development['label'] != 'valid_ppl':
        raise ValueError('The milestone report requires development validation-PPL results')
    build_report(args.output, development, mechanism)


if __name__ == '__main__':
    main()
