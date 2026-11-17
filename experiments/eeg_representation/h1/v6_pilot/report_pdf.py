"""Self-contained PDF with measured layout and every-page Poppler rendering."""
from pathlib import Path
import json
import math
import os
import subprocess
import tempfile
from xml.sax.saxutils import escape
import numpy as np
from .reporting import GLOSSARY, _display, _hash, _json, _number, _anchor_acquisition, _preferred_population

PAGE_W,PAGE_H=595.276,841.890
LEFT,RIGHT,TOP,BOTTOM=42.,42.,62.,43.
WIDTH=PAGE_W-LEFT-RIGHT


def _safe(text):
    return str(text).replace('\u2011','-').replace('\u2013','-').replace('\u2014',' - ').replace('\u2212','-').replace('\u00b2','^2').replace('\u00b5','u').replace('\u2264','<=').replace('\u2265','>=')


class Document:
    def __init__(self,path,engineering=False):
        from reportlab.pdfgen.canvas import Canvas
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import matplotlib
        fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
        for name,filename in [('PilotSans','DejaVuSans.ttf'),('PilotSans-Bold','DejaVuSans-Bold.ttf')]:
            if name not in pdfmetrics.getRegisteredFontNames():pdfmetrics.registerFont(TTFont(name,str(fontdir/filename)))
        self.canvas=Canvas(str(path),pagesize=(PAGE_W,PAGE_H),pageCompression=1)
        self.canvas.setTitle('NeuroStimEnv: exploratory reporting pilot')
        self.canvas.setAuthor('NeuroStimEnv artifact-backed reporting')
        self.style=ParagraphStyle('body',fontName='PilotSans',fontSize=9,leading=12.5,textColor=HexColor('#202d38'),spaceAfter=5)
        self.small=ParagraphStyle('small',parent=self.style,fontSize=7.3,leading=9.5)
        self.heading=ParagraphStyle('heading',parent=self.style,fontName='PilotSans-Bold',fontSize=16,leading=19)
        self.subheading=ParagraphStyle('subheading',parent=self.style,fontName='PilotSans-Bold',fontSize=10.5,leading=14)
        self.page=0;self.y=PAGE_H-TOP;self.items=[];self.engineering=engineering;self.new_page()

    def new_page(self):
        from reportlab.lib.colors import HexColor
        if self.page:self.canvas.showPage()
        self.page+=1;self.y=PAGE_H-TOP
        self.canvas.setFillColor(HexColor('#143A52'));self.canvas.rect(0,PAGE_H-29,PAGE_W,29,fill=1,stroke=0)
        self.canvas.setFillColor('white');self.canvas.setFont('PilotSans-Bold',8.5)
        self.canvas.drawString(LEFT,PAGE_H-19,'NEUROSTIMENV  |  REPORTING PILOT')
        self.canvas.setFillColor(HexColor('#5a6770'));self.canvas.setFont('PilotSans',7)
        self.canvas.drawString(LEFT,23,'ENGINEERING TEMPLATE - NOT EXPERIMENTAL RESULTS' if self.engineering else 'Exploratory inspected-data pilot. Confirmation deferred.')
        self.canvas.drawRightString(PAGE_W-RIGHT,23,str(self.page))

    def ensure(self,height):
        if height>PAGE_H-TOP-BOTTOM:raise ValueError('Single PDF element exceeds page height')
        if self.y-height<BOTTOM:self.new_page()

    def record(self,kind,x,y,w,h,font=None):
        self.items.append({'page':self.page,'kind':kind,'x':x,'y':y,'width':w,'height':h,'font_size':font})

    def para(self,text,kind='body',gap=6):
        from reportlab.platypus import Paragraph
        style={'body':self.style,'small':self.small,'heading':self.heading,'subheading':self.subheading}[kind]
        # Pass only escaped text; no uncontrolled markup from evidence files.
        if kind in ('heading','subheading'):self.ensure(65)
        p=Paragraph(escape(_safe(text)).replace('\n','<br/>'),style);w,h=p.wrap(WIDTH,10000)
        if h>PAGE_H-TOP-BOTTOM:
            chunks=p.split(WIDTH,PAGE_H-TOP-BOTTOM)
            if not chunks:raise ValueError('Unbreakable paragraph')
            for item in chunks:
                _,height=item.wrap(WIDTH,10000);self.ensure(height+gap);item.drawOn(self.canvas,LEFT,self.y-height);self.record('paragraph',LEFT,self.y-height,WIDTH,height,style.fontSize);self.y-=height+gap
            return
        self.ensure(h+gap);p.drawOn(self.canvas,LEFT,self.y-h);self.record('paragraph',LEFT,self.y-h,WIDTH,h,style.fontSize);self.y-=h+gap

    def section(self,title,new=False):
        if new and self.y<PAGE_H-TOP-10:self.new_page()
        self.para(title,'heading',12)

    def table(self,headers,rows,widths=None):
        from reportlab.platypus import Paragraph
        from reportlab.lib.colors import HexColor
        weights=np.asarray(widths if widths else [1]*len(headers),dtype=float);widths=WIDTH*weights/weights.sum()
        def draw(row,header=False):
            cells=[Paragraph(escape(_safe(_display(v))).replace('\n','<br/>'),self.small) for v in row]
            heights=[cell.wrap(float(width)-10,10000)[1] for cell,width in zip(cells,widths)];height=max(heights)+11
            if height>PAGE_H-TOP-BOTTOM:raise ValueError('PDF table row exceeds page')
            if self.y-height<BOTTOM:
                self.new_page()
                if not header:draw(headers,True)
            x=LEFT; y=self.y-height
            self.canvas.setFillColor(HexColor('#e5eef4' if header else '#f8fafb'))
            self.canvas.rect(x,y,WIDTH,height,fill=1,stroke=0)
            self.canvas.setStrokeColor(HexColor('#d8e0e5'));self.canvas.line(x,y,x+WIDTH,y)
            for cell,width,ch in zip(cells,widths,heights):cell.drawOn(self.canvas,x+5,self.y-5-ch);x+=width
            self.record('table_header' if header else 'table_row',LEFT,y,WIDTH,height,self.small.fontSize);self.y=y
        draw(headers,True)
        if rows:
            for row in rows:draw(row)
        else:draw(['Unavailable: no measured records']+['']*(len(headers)-1))
        self.y-=10

    def figure(self,path,caption):
        from PIL import Image
        with Image.open(path) as image:iw,ih=image.size
        width=WIDTH;height=min(290.,WIDTH*ih/iw)
        width=height*iw/ih
        # Keep the short caption with its image whenever possible.
        from reportlab.platypus import Paragraph
        p=Paragraph(escape(_safe(caption)),self.small);_,ch=p.wrap(WIDTH,10000)
        self.ensure(height+ch+15)
        x=LEFT+(WIDTH-width)/2;y=self.y-height
        self.canvas.drawImage(str(path),x,y,width,height,mask='auto');self.record('figure',x,y,width,height)
        self.y=y-5;self.para(caption,'small',10)

    def finish(self):
        self.canvas.save()
        violations=[]
        for item in self.items:
            if item['x']<LEFT-.1 or item['y']<BOTTOM-.1 or item['x']+item['width']>PAGE_W-RIGHT+.1 or item['y']+item['height']>PAGE_H-TOP+.1:violations.append(item)
        return {'pages':self.page,'page_size_pt':[PAGE_W,PAGE_H],'content_bounds_pt':[LEFT,BOTTOM,PAGE_W-RIGHT,PAGE_H-TOP],
            'items':self.items,'minimum_body_font_pt':7.3,'boundary_violations':violations,
            'status':'complete' if not violations else 'failed',
            'method':'Paragraph wrapping and measured row/image extents recorded at draw time. Headers/footers are outside the declared content frame by design.'}


def _counts(row):
    if not any(row.get(k) is not None for k in ('circuits','contexts','futures')):return 'Not applicable'
    return '/'.join(_display(row.get(k)) for k in ('circuits','contexts','futures'))


def _uncertainty(row):
    if row.get('evidence_type')=='theoretical':return 'Not empirical'
    if row.get('evidence_type')=='unknown':return 'Unknown ideal risk'
    value=row.get('uncertainty','Deferred: pilot precision')
    if isinstance(value,dict):
        if value.get('status') in ('deferred','unavailable'):return str(value.get('status')).capitalize()+': '+str(value.get('reason','limited pilot precision'))
        if all(_number(value.get(k)) for k in ('lower','upper')):
            return f"{100*value.get('confidence',.95):g}% [{value['lower']:.5g}, {value['upper']:.5g}]; descriptive circuit interval"
        if 'mean_error_interval' in value:
            interval=value.get('mean_error_interval'); text='Mean: '+_display(interval)
            text+='; squared variance error: '+('interval deferred' if value.get('variance_error_interval') is None else _display(value['variance_error_interval']))
            return text
    return _display(value)


def _summary_items(mapping, prefix='', depth=0):
    """Readable scalar ledger; full structured records remain lossless in JSON."""
    rows=[]
    for key,value in mapping.items():
        label=(prefix+' / ' if prefix else '')+str(key).replace('_',' ')
        if isinstance(value,dict):
            if depth<3:rows.extend(_summary_items(value,label,depth+1))
            else:rows.append([label,f'{len(value)} keyed records; full values in report_input.json'])
        elif isinstance(value,list):rows.append([label,f'{len(value)} records; full values in report_input.json'])
        else:rows.append([label,_display(value)])
    return rows


def create_pdf(out,results,checks,figures):
    out=Path(out);tmp=out/'pilot_findings.pdf.tmp';doc=Document(tmp,results.get('purpose')=='engineering_fixture')
    fm={r['name']:r for r in figures}
    def figure(name):
        if name in fm:doc.figure(out/fm[name]['paths']['png'],fm[name]['caption'])
        else:doc.para('Figure unavailable: '+name+'. No outcomes were fabricated to fill this panel.','small')
    doc.section('EEG information, prediction and task retention')
    doc.para('A bounded exploratory reporting pilot','subheading')
    doc.para('Purpose: finalize understandable manuscript metrics, tables and figures before deciding on the large confirmation study. These inspected toy-model circuits are not human data, a powered confirmation or evidence of objective superiority.')
    if results.get('purpose')=='engineering_fixture':doc.para('ENGINEERING FIXTURE: all displayed numbers exercise reporting software. They are not simulator findings, prospective outcomes or a biological experiment.','subheading')
    doc.para('Experimental completion: '+checks['status']+'. Report completion and scientific completion are separate checks. No automatic power-based sample expansion is part of this pilot.')
    allocation=results.get('completed_allocation',{});runtime=results.get('runtime',{})
    doc.table(['Allocation / execution fact','Recorded value'],_summary_items({'allocation':allocation,'runtime':runtime}),[1.1,2.5])
    doc.para('Eight workers means eight NEURON simulation processes, each restricted to one CPU thread. GPU fitting is separately timed. Workers, repeated futures and optimization repeats do not create independent circuits. Missing measured runtimes remain unavailable.')
    findings=results['tables']['findings']
    doc.para('Observed findings','subheading')
    if findings:
        for row in findings[:5]:
            doc.para(_display(row.get('question'))+': '+_display(row.get('observed'))+'. '+_display(row.get('interpretation'))+' Limitation: '+_display(row.get('limitation')))
    else:doc.para('No completed finding records are available. This document preserves missing stages and is not a completion receipt.')
    doc.section('Design, state coverage and model verification',new=True)
    doc.para('Ordinary EEG and code analyses reuse historical train/development/evaluation splits and are explicitly already inspected. The smaller state-recovery cohort, two-context H2 extension and privileged replay are reported separately. A learner comparison must match target, population, normalization and scoring futures.')
    state=results.get('complete_state_verification',{})
    doc.table(['Required check','Observed status'],[[k.replace('_',' '),_display(v)] for k,v in state.items() if k!='artifact_manifest'],[1.5,2.5])
    coverage=results.get('coverage',{})
    inv=coverage.get('inventory',[]) if isinstance(coverage,dict) else []
    if inv:
        doc.para('Present-state field coverage','subheading')
        doc.table(['Field / mechanism','Units / shape / identity','Coverage'],[[_display(r.get('field',r.get('name'))),_display(r.get('units'))+'; '+_display(r.get('shape'))+'; '+_display(r.get('identity')), _display(r.get('status'))] for r in inv],[1.3,2.3,1])
    else:
        doc.table(['Coverage audit item','Recorded value'],_summary_items(coverage),[1.6,3])
        if isinstance(coverage,dict) and coverage.get('fields'):
            audits=coverage['fields'];summaries=[]
            for key in ('field_count','numeric_coordinate_count'):
                vals=[a[key] for a in audits if key in a]
                if vals:summaries.append([key.replace('_',' '),str(min(vals))+' to '+str(max(vals))])
            for key in ('n_cells','segment_count','contact_count','external_connections','recurrent_connections','pending_recurrent_count','extracellular_layers','source_detector_count'):
                vals=[a.get('coverage',{}).get(key) for a in audits];vals=[v for v in vals if isinstance(v,(int,float))]
                if vals:summaries.append([key.replace('_',' '),str(min(vals))+' to '+str(max(vals))])
            doc.table(['Actual per-context coverage','Observed range across saved exports'],summaries,[1.7,2.5])
        doc.para('Every per-context field inventory and source binding is retained in report_input.json and the complete-state schema. Lists are summarized by count here; no input coordinates are removed from the learner.','small')
    doc.para('Required physical inputs include segment voltages/extracellular state, mechanism gates, Exp2Syn A/B, pre-cutoff-caused pending deliveries, current drive state, connectivity, parameters, geometry/forward transform and needed acquisition state. Future external arrival times, future randomness and equivalent encodings are excluded. A full-prefix replay is not a checkpoint-restoration claim, and replay success alone cannot prove adapter completeness.')
    doc.para('Declared learner policy and tuning budget','subheading')
    doc.table(['Configuration item','Declared value'],_summary_items(results.get('learner_protocol',{})),[1.6,3])
    doc.para('The fixed ridge grid is development-selected; the nonmean state candidate is retained separately. The new neural demonstration uses one optimization seed and at most 500 updates with development early stopping on A1 h0/h10 W128 and two histories. A reused forecast encoder has its original larger pretraining cohort, whereas projected ranges and downstream readouts use the labelled pilot cohort. Declared budgets are not proof that an unrun fit completed.','small')
    doc.para('A completed pilot must retain an actually trained, state-dependent candidate even if development selection prefers the mean fallback. Complete-state fitting can fail statistically in a small cohort; its measured risk is never replaced with the repeated-future variance or theoretical zero.')
    doc.section('Target timing and metric glossary',new=True);figure('target_timing')
    for name,body in GLOSSARY:doc.para(name,'subheading');doc.para(body)
    doc.section('H1 | Same state, several possible futures',new=True);figure('same_state_futures')
    doc.para('Question: how much future variation persists from the same declared physical present? Conditional future variability and across-circuit variability answer different questions. The empirical mean/spread are finite-ensemble summaries and do not establish an EEG-only minimum error.')
    doc.section('Table 1 | Information available versus future error',new=True)
    doc.para('Main panels: immediate h0/W128 and delayed h10/W128. Other settings remain in the 60-setting appendix and unrounded tables. Model risk minus estimated state variance is a combined unresolved excess, not separately measured observation, compression and fitting gaps.')
    anchor_acq=_anchor_acquisition(results['tables']['h1_information'])
    doc.para('Displayed acquisition: '+anchor_acq+'. Absent anchors are unavailable; they are not substituted with another target.','small')
    main=[r for r in results['tables']['h1_information'] if r.get('h_ms') in (0,10) and r.get('duration_ms')==128 and r.get('acquisition')==anchor_acq]
    # Separate populations explicitly; never insert a replay number into the ordinary ladder.
    main_population=_preferred_population(main)
    main=[r for r in main if r.get('population')==main_population or r.get('evidence_type') in ('theoretical','unknown') or 'replay' in str(r.get('population','')).lower()]
    for pop in sorted({r.get('population','unspecified') for r in main}):
        doc.para('Population: '+str(pop),'subheading')
        rows=[r for r in main if r.get('population','unspecified')==pop]
        doc.table(['h / information; method','Mean=100 error','Descriptive uncertainty','Circuits / contexts / M','Evidence'],[[str(r.get('h_ms'))+' / '+str(r.get('information',r.get('method')))+'; '+str(r.get('method')),r.get('error_baseline100'),_uncertainty(r),_counts(r),str(r.get('evidence_type',''))+'; '+str(r.get('status'))] for r in rows],[2.7,1,1.7,1,1.3])
    if not main:doc.para('No measured main-anchor rows available. Complete-state selected and state-dependent candidate rows remain required for experimental completion.')
    figure('information_error');figure('privileged_replay')
    doc.section('Table 2 | Immediate, delayed and history comparisons',new=True)
    doc.para('Question: does a fitted predictor benefit from a shorter start offset, a longer observed history or a changed acquisition? Compare acquisitions only at the same target and baseline. Changes in fitted-model performance do not by themselves isolate information loss.')
    rows=[r for r in results['tables']['h1_comparisons'] if r.get('population')==_preferred_population(results['tables']['h1_comparisons'])]
    doc.para('Main comparison population: '+str(_preferred_population(rows))+'. Broader historical results are separate in the appendix and companion tables.','small')
    doc.table(['Acq / h / W / history','Mean=100 error','MSE reduction (pp)','Paired difference / uncertainty'],[[str(r.get('acquisition'))+' / '+str(r.get('h_ms'))+' / '+str(r.get('duration_ms'))+' / '+str(r.get('history')),r.get('error_baseline100'),r.get('relative_mse_reduction_pp'),_display(r.get('paired_difference'))+'; '+_uncertainty(r)] for r in rows],[2,1,1.2,2.5])
    figure('gap_history_acquisition')
    doc.section('Table 3 | H2 finite-future estimation',new=True)
    doc.para('Question: how does conditional mean/variance estimation change as M grows? The default pilot adds 24 independent scoring futures to each of two fixed legacy contexts, reusing disjoint L32 references. M=1/2/4/8/16/32 are nested; repeated subsets/reference use are dependent. This is not the deferred L128 design. A separate fresh fallback population must be labelled if legacy continuation reproduction fails.')
    rows=results['tables']['h2_precision'];chosen=next((r.get('job') for r in rows if str(r.get('job')).startswith('A1') and 'l0_d128' in str(r.get('job'))),rows[0].get('job') if rows else None)
    rr=[r for r in rows if r.get('job')==chosen];doc.para('Main target: '+(str(chosen) if chosen is not None else 'unavailable')+'. All target-specific H2 records are in the companion table.','small')
    if rr:doc.para('Mean-error units: '+str(rr[0].get('mean_error_units','unavailable'))+'. Variance-error units: '+str(rr[0].get('variance_error_units','unavailable'))+'. Variance error is squared variance-estimation error. An interval for a signed variance difference is not an interval for this squared error.','small')
    doc.table(['Context / M / L','Corrected mean error','Variance error','Uncertainty / reference accounting'],[[str(r.get('context_id'))+' / '+str(r.get('M'))+' / '+str(r.get('L')),r.get('mean_error_corrected'),r.get('variance_error'),_uncertainty(r)+'; '+_display(r.get('reference_uncertainty','Unavailable'))] for r in rr],[2.1,1.2,1.2,2.5])
    figure('h2_finite_futures')
    doc.section('Table 4 | H3 reconstruction versus usefulness',new=True)
    doc.para('Question: do the same codes preserve past waveform reconstruction, future prediction and carrier classification equally well? d8/B32 is eight coordinates at four bits each, 32 packed payload bits. Projected-unquantized controls separate quantization from projection; shared model/codebook storage is additional. Any differing upstream training must remain explicit.')
    doc.table(['Representation / budget','Past NMSE','Future mean=100','Carrier CE bits / accuracy','Cohort / uncertainty'],[[str(r.get('representation'))+' / d8 / '+('B32' if r.get('budget_bits')==32 else 'projected unquantized'),r.get('reconstruction_nmse'),r.get('future_error_baseline100'),_display(r.get('carrier_ce_bits'))+' / '+_display(r.get('accuracy')),_display(r.get('population'))+'; '+_uncertainty(r)] for r in results['tables']['h3_retention']],[2,1,1,1.4,2])
    figure('h3_reconstruction_usefulness')
    doc.section('Table 5 | H4 target definition and physical support',new=True)
    doc.para('Question: how does the prediction problem change with the target? Waveform, future-only Fourier coefficients, direct energy and log energy are different estimands. A one-second block is available after its full second, not 10 ms ahead. Expected band energy is generally not the energy of the predicted conditional-mean waveform. Compare each method to the matching training-only reference.')
    h4_acq='A1' if any(r.get('acquisition')=='A1' for r in results['tables']['h4_targets']) else anchor_acq
    rows=[r for r in results['tables']['h4_targets'] if r.get('acquisition','A1')==h4_acq and r.get('population')==_preferred_population(results['tables']['h4_targets'])]
    doc.para('Acquisition: '+h4_acq+'. Main population: '+str(_preferred_population(rows))+'.','small')
    doc.table(['Target / h / W','Target units; risk units','State var / ref','MSE reduction pp','Uncertainty'],[[str(r.get('target'))+' / '+str(r.get('h_ms'))+' / '+str(r.get('duration_ms')),_display(r.get('physical_units'))+'; '+_display(r.get('risk_units')),r.get('floor_reference_ratio'),r.get('relative_mse_reduction_pp'),_uncertainty(r)] for r in rows],[2.2,1.5,1,1,1.5])
    figure('h4_target_outcomes')
    doc.section('Run status, limitations and review decisions',new=True)
    doc.table(['Stage / evidence category','Status / count','Details'],[[str(r.get('stage'))+' / '+str(r.get('category')),str(r.get('status'))+' / '+_display(r.get('count')),r.get('details')] for r in results['tables']['run_status']],[1.5,1,3])
    doc.para('Reporting changes to review before confirmation','subheading')
    for s in ('Check whether state coverage, actual training and held-out matching are documented sufficiently to justify the complete-state label.',
              'Check every denominator, unit, target timestamp and population label; past NMSE, future mean=100 error and task CE must remain distinct.',
              'Review negative/inconclusive results without interpreting a fitted learner as an information oracle.',
              'Review all rendered PDF pages and the complete 60-setting appendix. Descriptive uncertainty must state its unit and unavailable fields.',
              'Decide separately whether a later powered/frozen confirmation is warranted. This pilot never launches it.'):
        doc.para(s)
    if checks['issues']:
        doc.para('Experimental completion blockers','subheading')
        for issue in checks['issues'][:25]:doc.para(issue,'small')
        if len(checks['issues'])>25:doc.para(f'{len(checks["issues"])-25} additional exact blockers are in report/receipt.json.','small')
    doc.section('Appendix | Complete acquisition-target matrix',new=True)
    doc.para('One compact row per configured setting; full information ladders, H2 records and all control readouts are in the editable companion tables. Physical MSE and reference risk are stored without display rounding in JSON/NPZ/CSV. An unavailable value is not zero. The ordinary historical eight-circuit cohort remains distinct from the smaller matched state cohort.')
    rows=results['tables']['full_grid']
    if not rows:
        rows=results['tables']['h4_targets']
    doc.table(['Acquisition / target / support','Method / population','Mean=100 / reduction pp','Status'],[[str(r.get('job',str(r.get('acquisition'))+' / '+str(r.get('target'))))+('; h'+_display(r.get('h_ms'))+' W'+_display(r.get('duration_ms')) if r.get('h_ms') is not None else ''),_display(r.get('method','selected'))+'; '+str(r.get('population','unavailable')),_display(r.get('error_baseline100'))+' / '+_display(r.get('relative_mse_reduction_pp')),r.get('status')] for r in rows],[2.4,2,1.3,1])
    broad=[r for r in results['tables']['h4_targets'] if 'historical' in str(r.get('population','')).lower() and r.get('population')!=_preferred_population(results['tables']['h4_targets'])]
    if broad:
        doc.section('Appendix | Broader historical eight-circuit results',new=True)
        doc.para('Separate inspected historical population; these rows are not substituted into the two-circuit matched state ladder. Physical risks, code methods and all controls remain in the companion tables.')
        doc.table(['Acquisition / target','Mean=100 error','MSE reduction (pp)','Uncertainty'],[[r.get('job'),r.get('error_baseline100'),r.get('relative_mse_reduction_pp'),_uncertainty(r)] for r in broad],[2.3,1,1,2.5])
    doc.para('Evidence provenance','subheading')
    doc.para('Source fingerprints, physical generation provenance, target/operator/schema hashes, exact input arrays, fitted predictions and report hashes are preserved in the lossless evidence export. This PDF is a rounded presentation of those records. Re-rendering a report does not rerun the experiment or change its scientific provenance.','small')
    layout=doc.finish();os.replace(tmp,out/'pilot_findings.pdf');_json(out/'pdf_layout.json',layout);return layout


def verify_pdf(out,layout,*,render=True):
    from pypdf import PdfReader
    from PIL import Image,ImageDraw
    out=Path(out);path=out/'pilot_findings.pdf';reader=PdfReader(path)
    errors=[];pages=[];renderdir=out/'rendered';renderdir.mkdir(exist_ok=True)
    if len(reader.pages)!=layout['pages']:errors.append('PDF page count differs from layout count')
    if layout['boundary_violations']:errors.append('Content crosses measured page frame')
    for i,page in enumerate(reader.pages,1):
        text=page.extract_text() or ''
        if len(text.strip())<30:errors.append('Insufficient text on page '+str(i))
        if '\u25a0' in text or '\ufffd' in text:errors.append('Replacement glyph in page '+str(i))
        pages.append({'page':i,'text_characters':len(text),'text_sha256':__import__('hashlib').sha256(text.encode()).hexdigest()})
    if render:
        # Remove only previews from our prior report; failed rebuilds cannot mix page sets.
        for p in renderdir.glob('page-*.png'):p.unlink()
        proc=subprocess.run(['pdftoppm','-r','100','-png',str(path),str(renderdir/'page')],capture_output=True,text=True)
        if proc.returncode:errors.append('Poppler render failed: '+proc.stderr[-1500:])
        images=sorted(renderdir.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))
        if len(images)!=len(reader.pages):errors.append('Not every PDF page rendered')
        thumbnails=[]
        for rec,img in zip(pages,images):
            with Image.open(img) as im:
                gray=np.asarray(im.convert('L'));ink=float(np.mean(gray<245));rec.update({'render':str(img.relative_to(out)),'render_sha256':_hash(img),'nonwhite_fraction':ink,'pixel_size':list(im.size)})
                if not .005<ink<.95:errors.append('Blank/invalid rendered page '+str(rec['page']))
                thumb=im.convert('RGB');thumb.thumbnail((238,337));thumbnails.append(thumb.copy())
        # Rendered contact sheets are proof each page exists, not a claim of human QA.
        for start in range(0,len(thumbnails),12):
            subset=thumbnails[start:start+12];sheet=Image.new('RGB',(4*258,math.ceil(len(subset)/4)*365),'#dbe2e6');draw=ImageDraw.Draw(sheet)
            for j,im in enumerate(subset):x=(j%4)*258+10;y=(j//4)*365+20;sheet.paste(im,(x,y));draw.text((x,y-15),'Page '+str(start+j+1),fill='black')
            sheet.save(renderdir/f'contact-sheet-{start//12+1:02d}.png')
    else:errors.append('Every-page rendering was disabled')
    result={'status':'complete' if not errors else 'failed','pages':pages,'errors':errors,'all_pages_rendered':render and all('render' in p for p in pages),
            'layout_boundary_check':layout['status'],'minimum_table_font_pt':7.3,'human_visual_review':'Not asserted by automated code; inspect rendered pages separately.',
            'automated_checks':'Every-page Poppler rendering, page/text count, replacement-glyph check, measured paragraph/table/image bounds, minimum typeset body font and blank/solid-render detection.'}
    _json(out/'pdf_verification.json',result);return result
