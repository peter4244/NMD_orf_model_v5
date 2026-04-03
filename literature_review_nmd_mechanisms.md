# NMD Mechanisms: Literature Review in Context of ORF Model Findings

## Summary

Our deep learning model for NMD prediction reveals three principal findings that connect to distinct bodies of NMD literature: (1) PTC + downstream EJC is the dominant predictor of NMD visibility, (2) approximately 15% of NMD isoforms appear driven by 5'UTR features consistent with aberrant translation initiation, and (3) 3'UTR length does not strongly predict NMD susceptibility in our system. We additionally propose that ribosomal ORF selection may be stochastic, with NMD susceptibility reflecting the interaction between start codon strength and the NMD potential of the engaged reading frame. Here we review how these findings relate to the current state of knowledge.

### 1. PTC + EJC as the dominant NMD mechanism

The model's reliance on downstream exon-junction complexes as the primary predictor of NMD is consistent with the well-established EJC model of mammalian NMD. The physical basis for this model was established by the discovery that splicing deposits a multiprotein complex 20-24 nucleotides upstream of exon-exon junctions [1], and the "50-55 nucleotide rule" — that a stop codon located more than 50-55 nt upstream of an exon-exon junction triggers NMD — has served as the operational definition of a PTC in mammalian systems since its articulation by Nagy and Maquat [2]. The mechanistic pathway from PTC recognition to mRNA degradation proceeds through the SURF complex (SMG1-UPF1-eRF1-eRF3), which assembles at the terminating ribosome and interacts with downstream EJC components to form the DECID complex, committing the transcript to degradation via UPF1 phosphorylation [3, 4]. Our experimental design using SMG1 inhibition directly targets this commitment step.

The EJC-dependent pathway is broadly recognized as the dominant NMD mechanism in mammals [5, 6, 7], though it is not the only one. Comparative studies of EJC-dependent and EJC-independent NMD have found the two pathways to be partially redundant, with EJC-enhanced NMD predominating [8]. Our model's strong weighting of the `n_downstream_ejc` structural feature (the single most important predictor for PTC+ isoforms) and the dose-response relationship between EJC count and prediction probability quantitatively recapitulate this consensus. Importantly, our model also captures the graded nature of NMD — consistent with observations that NMD efficiency varies across transcripts and is not a simple binary switch [9, 10].

### 2. 5'UTR-mediated NMD and the "confused ribosome"

Approximately 15% of NMD isoforms in our data lack the canonical PTC + EJC signature, and the model shifts to using start codon context and 5'UTR composition features to predict their NMD status. This subpopulation is consistent with the well-documented class of uORF-containing NMD targets first characterized at scale by Mendell et al. [11], who showed that NMD regulates 5-10% of the transcriptome including many transcripts without obvious PTCs, with uORF-containing mRNAs forming a major class. Transcriptome-wide UPF1 binding studies confirmed that NMD factors associate with uORF-containing transcripts [12], and several reviews have characterized uORF-mediated NMD as a bona fide gene-regulatory mechanism [13, 14, 15].

The mechanistic basis for uORF-mediated NMD involves translation reinitiation. After translating a short uORF, ribosomes can reinitiate at downstream AUGs, but reinitiation efficiency depends on uORF length, inter-cistronic distance, and Kozak context [16]. When reinitiation occurs at a suboptimal position — for example, at an out-of-frame AUG or at a position that places the eventual stop codon upstream of an EJC — the transcript becomes an NMD target. Our model's elevated ATG branch importance for the PTC- ref ATG retained subgroup, combined with its diffuse attention across multiple ORFs, is consistent with this mechanism: the model searches the 5'UTR landscape for signals of aberrant translation initiation rather than relying on a single canonical start codon.

We describe this pattern informally as the "confused ribosome" hypothesis — that NMD sensitivity in these isoforms arises from uncertainty in ORF selection at the translation initiation stage. While this specific framing is not an established term in the literature, its components are well-supported: the role of Kozak context in modulating start site selection [17, 18], the prevalence of uORFs as translational regulators [15, 19], and the regulatory coupling between translation initiation and NMD [14] are all well-documented. The novelty of our contribution lies in the model's data-driven discovery that these features form a coherent alternative prediction strategy, distinct from the canonical PTC pathway, in a measurable subpopulation of NMD targets.

### 3. 3'UTR length does not strongly predict NMD

Our analyses provide strong evidence that 3'UTR length does not independently drive NMD susceptibility in our lung cell system. In the isopair analysis (Section 3b of the isoform transition report), three lines of evidence converge: (1) PTC-negative NMD comparators have 3'UTR lengths indistinguishable from Control isoforms (p = 0.37, corrected for PTC-induced 3'UTR inflation by measuring from the reference stop codon position); (2) a prediction model showed that adding 3'UTR length to the full model does not improve holdout AUC (0.896 to 0.897), meaning 3'UTR length carries no predictive information beyond downstream EJC count; and (3) 3'UTR splicing is depleted in PTC-negative NMD comparators relative to Control (OR = 0.71, p = 2.63e-8), while enriched in PTC-positive comparators (OR = 1.78), consistent with 3'UTR splicing operating as a PTC-creating mechanism (via EJC repositioning) rather than an independent NMD trigger. Among same-stop PTC+ pairs — where the comparator shares the reference stop codon but has gained 3'UTR splice junctions — 86% have more downstream EJCs in the comparator, confirming the EJC repositioning mechanism.

These findings engage the "faux 3'UTR" model, originally proposed by Amrani et al. [20] in yeast, which posits that when a stop codon is far from the poly(A) tail, the resulting 3'UTR is perceived as aberrant and triggers NMD. In yeast, which lacks EJCs, this distance-dependent mechanism is the primary NMD trigger. In mammals, the picture is more nuanced. Several studies have demonstrated that long 3'UTRs can trigger NMD in mammalian cells [21], and that UPF1 binds 3'UTRs in a length-dependent manner [22]. The competition model proposes that NMD is determined by the balance between UPF1/EJC complexes (pro-NMD) and PABP proximity to the terminating ribosome (anti-NMD), with longer 3'UTRs tipping the balance toward degradation [23, 24]. However, several lines of evidence suggest this pathway plays a limited role in mammalian steady-state NMD. Many long 3'UTRs contain protective elements — including PTBP1 binding sites and structured RNA elements — that actively prevent NMD [25, 26]. A growing consensus holds that the faux 3'UTR model is more relevant in yeast than in mammals, where the EJC pathway dominates [27].

Our experimental data therefore align well with the emerging view: in the context of endogenous human isoform diversity, 3'UTR length is not an independent NMD trigger. When 3'UTR splicing does contribute to NMD, it does so by repositioning EJCs to create PTCs — reinforcing rather than replacing the canonical EJC-dependent pathway.

### 4. Stochastic ORF selection and NMD susceptibility

We hypothesize that ribosomal ORF selection may be partially stochastic, such that NMD susceptibility of a transcript is a function of start codon strength (Kozak context) combined with the "NMD potential" of the downstream reading frame. This hypothesis is supported by our model's multi-ORF architecture, in which the learned attention mechanism distributes weight across up to 5 candidate ORFs per transcript — particularly for non-canonical NMD isoforms where attention entropy is high.

The mechanistic foundation for stochastic ORF selection is the leaky scanning model: the 43S preinitiation complex can bypass AUGs in suboptimal Kozak context, allowing ribosomes to initiate at downstream start codons [17, 18, 28]. Ribosome profiling studies have provided direct evidence that single transcripts engage ribosomes at multiple positions. The foundational work by Ingolia et al. [29, 30] revealed widespread translation of uORFs and alternative ORFs, and specialized initiation-site profiling (QTI-seq) confirmed that most mRNAs have multiple active start codons with usage frequencies that correlate with — but are not fully determined by — Kozak context strength [31, 32]. Cross-species analyses have shown that uORF-mediated translational repression is conserved and quantitatively related to Kozak context [33, 34].

The connection between stochastic initiation and NMD has been demonstrated in specific systems: altering translation initiation efficiency affects NMD susceptibility [35], and ribosome occupancy at uORF termination codons directly modulates NMD of downstream transcripts [36]. Our Kozak PWM analysis supports this framework: PTC+ isoforms have the strongest Kozak context at their reference CDS start codons (mean PWM = 0.79), while PTC- ref ATG retained isoforms — despite having the same reference CDS ORF — show weaker Kozak context (mean PWM = 0.30), potentially allowing more ribosomal read-through to alternative ORFs with different NMD fates.

The specific formulation that NMD susceptibility should be modeled as a probability-weighted function over multiple candidate ORFs — where the probability is governed by Kozak context strength and the NMD potential by the ORF's termination context — appears to be a novel analytical framework. While the individual components are well-established, we are not aware of prior work that has explicitly modeled this interaction in a multi-ORF deep learning architecture or demonstrated its predictive utility at transcriptome scale.

## References

1. Le Hir H, Izaurralde E, Maquat LE, Moore MJ. The spliceosome deposits multiple proteins 20-24 nucleotides upstream of mRNA exon-exon junctions. *EMBO J.* 2000;19(24):6860-6869.

2. Nagy E, Maquat LE. A rule for termination-codon position within intron-containing genes: when nonsense affects RNA abundance. *Trends Biochem Sci.* 1998;23(6):198-199.

3. Kashima I, Yamashita A, Izumi N, et al. Binding of a novel SMG-1-Upf1-eRF1-eRF3 complex (SURF) to the exon junction complex triggers Upf1 phosphorylation and nonsense-mediated mRNA decay. *Genes Dev.* 2006;20(3):355-367.

4. Isken O, Maquat LE. Quality control of eukaryotic mRNA: safeguarding cells from abnormal mRNA function. *Genes Dev.* 2007;21(15):1833-3856.

5. Kurosaki T, Popp MW, Maquat LE. Quality and quantity control of gene expression by nonsense-mediated mRNA decay. *Nat Rev Mol Cell Biol.* 2019;20(7):406-420.

6. Popp MW, Maquat LE. Organizing principles of mammalian nonsense-mediated mRNA decay. *Annu Rev Genet.* 2013;47:139-165.

7. Kurosaki T, Maquat LE. Nonsense-mediated mRNA decay in humans at a glance. *J Cell Sci.* 2016;129(3):461-467.

8. Metze S, Herzog VA, Ruepp MD, Muhlemann O. Comparison of EJC-enhanced and EJC-independent NMD in human cells reveals two partially redundant degradation pathways. *RNA.* 2013;19(10):1432-1448.

9. Bhatt DM, Pandya-Jones A, Tong AJ, et al. Transcript dynamics of proinflammatory genes reveal that p65 mediates a revised NMD pathway. *Mol Cell.* 2012;46(5):585-595.

10. Karousis ED, Nasif S, Muhlemann O. Nonsense-mediated mRNA decay: novel mechanistic insights and relevant clinical implications. *Biol Chem.* 2016;397(11):1093-1126.

11. Mendell JT, Sharifi NA, Meyers JL, Martinez-Murillo F, Dietz HC. Nonsense surveillance regulates expression of diverse classes of mammalian transcripts and mutes genomic noise. *Nat Genet.* 2004;36(10):1073-1078.

12. Hurt JA, Robertson AD, Burge CB. Global analyses of UPF1 binding and function reveal expanded scope of nonsense-mediated mRNA decay. *Genome Res.* 2013;23(10):1636-1650.

13. Nickless A, Bailis JM, You Z. Control of gene expression through the nonsense-mediated RNA decay pathway. *Cell Biosci.* 2017;7:26.

14. Barbosa C, Peixeiro I, Romao L. Gene expression regulation by upstream open reading frames and human disease. *PLoS Genet.* 2013;9(8):e1003529.

15. Somers J, Poyry T, Willis AE. A perspective on mammalian upstream open reading frame function. *Int J Biochem Cell Biol.* 2013;45(8):1690-1700.

16. Kozak M. Constraints on reinitiation of translation in mammals. *Nucleic Acids Res.* 2001;29(24):5226-5232.

17. Kozak M. Point mutations define a sequence flanking the AUG initiator codon that modulates translation by eukaryotic ribosomes. *Cell.* 1986;44(2):283-292.

18. Kozak M. Pushing the limits of the scanning mechanism for initiation of translation. *Gene.* 2002;299(1-2):1-34.

19. Yepiskoposyan H, Aeschimann F, Nilsson D, Okoniewski M, Muhlemann O. Autoregulation of the nonsense-mediated mRNA decay pathway in human cells. *RNA.* 2011;17(12):2108-2118.

20. Amrani N, Ganesan R, Kerber S, Ghosh S, Jacobson A. A faux 3'-UTR promotes aberrant termination and triggers nonsense-mediated mRNA decay. *Nature.* 2004;432(7013):112-118.

21. Buhler M, Steiner S, Mohn F, Paillusson A, Muhlemann O. EJC-independent degradation of nonsense immunoglobulin-mu mRNA depends on 3'UTR length. *Nat Struct Mol Biol.* 2006;13(5):462-464.

22. Hogg JR, Goff SP. Upf1 senses 3'UTR length to potentiate mRNA decay. *Cell.* 2010;143(3):379-389.

23. Eberle AB, Stalder L, Mathys H, Orozco RZ, Muhlemann O. Posttranscriptional gene regulation by spatial rearrangement of the 3' untranslated region. *PLoS Biol.* 2008;6(4):e92.

24. Singh G, Rebbapragada I, Lykke-Andersen J. A competition between stimulators and antagonists of Upf complex recruitment governs human nonsense-mediated mRNA decay. *PLoS Biol.* 2008;6(4):e111.

25. Toma KG, Rebbapragada I, Durand S, Lykke-Andersen J. Identification of elements in human long 3'UTRs that inhibit nonsense-mediated decay. *RNA.* 2015;21(5):887-897.

26. Ge Z, Quek BL, Beemon KL, Hogg JR. Polypyrimidine tract binding protein 1 protects mRNAs from recognition by the nonsense-mediated mRNA decay pathway. *eLife.* 2016;5:e11155.

27. Boehm V, Haberman N, Hentze MW, Kulozik AE. Nonsense-mediated mRNA decay: the challenge of telling right from wrong in a complex transcriptome. *Wiley Interdiscip Rev RNA.* 2014;5(6):755-768.

28. Hinnebusch AG. The scanning mechanism of eukaryotic translation initiation. *Annu Rev Biochem.* 2014;83:779-812.

29. Ingolia NT, Ghaemmaghami S, Newman JRS, Weissman JS. Genome-wide analysis in vivo of translation with nucleotide resolution using ribosome profiling. *Science.* 2009;324(5924):218-223.

30. Ingolia NT, Lareau LF, Weissman JS. Ribosome profiling of mouse embryonic stem cells reveals the complexity and dynamics of mammalian proteomes. *Cell.* 2011;147(4):789-802.

31. Gao X, Wan J, Liu B, Ma M, Shen B, Qian SB. Quantitative profiling of initiating ribosomes in vivo. *Nat Methods.* 2015;12(2):147-153.

32. Lee S, Liu B, Lee S, Huang SX, Shen B, Qian SB. Global mapping of translation initiation sites in mammalian cells at single-nucleotide resolution. *Proc Natl Acad Sci USA.* 2012;109(37):E2424-E2432.

33. Johnstone TG, Bazzini AA, Giraldez AJ. Upstream ORFs are prevalent translational repressors in vertebrates. *EMBO J.* 2016;35(7):706-723.

34. Chew GL, Pauli A, Schier AF. Conservation of uORF repressiveness and sequence features in mouse, human, and zebrafish. *Nat Commun.* 2016;7:11663.

35. Kebaara BW, Atkin AL. Long 3'-UTRs target wild-type mRNAs for nonsense-mediated mRNA decay in Saccharomyces cerevisiae. *Nucleic Acids Res.* 2009;37(9):2771-2778.

36. Gaba A, Jacobson A, Sachs MS. Ribosome occupancy of the yeast CPA1 upstream open reading frame termination codon modulates nonsense-mediated mRNA decay. *Mol Cell.* 2005;20(5):747-758.

37. Hinnebusch AG, Ivanov IP, Sonenberg N. Translational control by 5'-untranslated regions of eukaryotic mRNAs. *Science.* 2016;352(6292):1413-1416.

38. Ferreira JP, Overton KW, Wang CL. Tuning gene expression with synthetic upstream open reading frames. *Proc Natl Acad Sci USA.* 2013;110(28):11284-11289.

39. Kozak M. Initiation of translation in prokaryotes and eukaryotes. *Gene.* 1999;234(2):187-208.

40. Lykke-Andersen J, Shu MD, Steitz JA. Communication of the position of exon-exon junctions to the mRNA surveillance machinery by the protein RNPS1. *Science.* 2001;293(5536):1836-1839.
