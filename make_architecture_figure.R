#!/usr/bin/env Rscript
# make_architecture_figure.R — v5 model architecture diagram (redesigned)
#
# Publication-quality figure showing:
#   0. Data preparation pipeline
#   1. Per-ORF encoding with visual input representations
#   2. K=5 ORF fan-out/fan-in through attention aggregation
#   3. Classification head
#   4. SHAP interpretation annotations at three model layers

library(ggplot2)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
col_atg      <- "#4ECDC4"   # teal
col_stop     <- "#FF6B6B"   # coral
col_struct   <- "#FFE66D"   # gold
col_fusion   <- "#95E1D3"   # light teal
col_attn     <- "#C3AED6"   # lavender
col_head     <- "#F38181"   # pink
col_shap     <- "#A8D8EA"   # light blue
col_arrow    <- "grey30"
col_prep     <- "#E8E8E8"   # light grey for data prep

# SHAP annotation colors
col_jds  <- "#2166AC"   # Joint DeepSHAP - blue
col_ks   <- "#B2182B"   # KernelSHAP - red
col_att  <- "#7B3294"   # Attention weights - purple

# Sequence channel colors (for visual strips)
ch_cols <- c("#109648", "#255C99", "#F7B32B", "#D62839",  # A, C, G, T
             "#888888", "#BBBBBB",                          # junction, GC
             "#9E7BB5", "#B99AD4", "#D4BDE8")               # frame 0, 1, 2

# Arrow helper
arr <- function(len = 0.3) arrow(length = unit(len, "cm"), type = "closed")

# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------
p <- ggplot() +
  coord_fixed(ratio = 1, xlim = c(-3, 17), ylim = c(-1, 20)) +
  theme_void() +
  theme(plot.margin = margin(5, 5, 5, 5),
        plot.background = element_rect(fill = "white", color = NA),
        panel.background = element_rect(fill = "white", color = NA))

# =========================================================================
# TITLE (y = 19-20)
# =========================================================================
p <- p +
  annotate("text", x = 7, y = 19.5, label = "v5 ORF-Centric Model Architecture",
           size = 9, fontface = "bold") +
  annotate("text", x = 7, y = 18.9, label = "From transcript sequence to NMD prediction",
           size = 6, color = "grey40")

# =========================================================================
# LEVEL 0: Data Preparation Pipeline (y = 16.5-18.0)
# =========================================================================
prep_y1 <- 16.4; prep_y2 <- 17.7; prep_ym <- (prep_y1 + prep_y2) / 2
prep_title_y <- prep_ym + 0.3; prep_detail_y <- prep_ym - 0.3

p <- p +
  # Box 1: Spliced mRNA
  annotate("rect", xmin = 0.0, xmax = 3.0, ymin = prep_y1, ymax = prep_y2,
           fill = col_prep, color = "grey50", linewidth = 0.8) +
  annotate("text", x = 1.5, y = prep_title_y, label = "Spliced mRNA",
           size = 6, fontface = "bold") +
  annotate("text", x = 1.5, y = prep_detail_y, label = "SQANTI FASTA",
           size = 5, color = "grey40") +

  # Arrow 1→2
  annotate("segment", x = 3.0, xend = 4.0, y = prep_ym, yend = prep_ym,
           arrow = arr(), color = col_arrow, linewidth = 0.8) +

  # Box 2: ORFik Scan
  annotate("rect", xmin = 4.0, xmax = 7.0, ymin = prep_y1, ymax = prep_y2,
           fill = col_prep, color = "grey50", linewidth = 0.8) +
  annotate("text", x = 5.5, y = prep_title_y, label = "ORFik Scan",
           size = 6, fontface = "bold") +
  annotate("text", x = 5.5, y = prep_detail_y, label = "All ORFs identified",
           size = 5, color = "grey40") +

  # Arrow 2→3
  annotate("segment", x = 7.0, xend = 8.0, y = prep_ym, yend = prep_ym,
           arrow = arr(), color = col_arrow, linewidth = 0.8) +

  # Box 3: Priority Selection
  annotate("rect", xmin = 8.0, xmax = 11.0, ymin = prep_y1, ymax = prep_y2,
           fill = col_prep, color = "grey50", linewidth = 0.8) +
  annotate("text", x = 9.5, y = prep_title_y, label = "Priority Selection",
           size = 6, fontface = "bold") +
  annotate("text", x = 9.5, y = prep_detail_y, label = "Ref CDS > TD2 CDS > Kozak",
           size = 4.5, color = "grey40") +

  # Arrow 3→4
  annotate("segment", x = 11.0, xend = 12.0, y = prep_ym, yend = prep_ym,
           arrow = arr(), color = col_arrow, linewidth = 0.8) +

  # Box 4: Per-ORF Encoding
  annotate("rect", xmin = 12.0, xmax = 15.0, ymin = prep_y1, ymax = prep_y2,
           fill = col_prep, color = "grey50", linewidth = 0.8) +
  annotate("text", x = 13.5, y = prep_title_y, label = "Per-ORF Encoding",
           size = 6, fontface = "bold") +
  annotate("text", x = 13.5, y = prep_detail_y, label = "9ch windows + 5 features",
           size = 5, color = "grey40")

# Level label
p <- p +
  annotate("text", x = -2.5, y = prep_ym, label = "0", size = 8,
           fontface = "bold", color = "grey60")

# =========================================================================
# FAN-OUT: 1 transcript → K=5 ORFs (y = 14.5-16.0)
# =========================================================================
# Arrow from data-prep center down
p <- p +
  annotate("segment", x = 7.5, xend = 7.5, y = prep_y1, yend = 15.9,
           arrow = arr(), color = col_arrow, linewidth = 0.8)

# 5 ORF pills
orf_xs <- c(1.5, 4.5, 7.5, 10.5, 13.5)
orf_w <- 1.2  # half-width
orf_y1 <- 14.8; orf_y2 <- 15.6

# Spread bar
p <- p +
  annotate("segment", x = orf_xs[1], xend = orf_xs[5], y = 15.9, yend = 15.9,
           color = "grey50", linewidth = 0.6)

for (i in seq_along(orf_xs)) {
  ox <- orf_xs[i]
  a <- if (i == 3) 0.9 else 0.4  # highlight center ORF
  p <- p +
    # Vertical stub from spread bar
    annotate("segment", x = ox, xend = ox, y = 15.9, yend = orf_y2,
             color = "grey50", linewidth = 0.6) +
    # ORF pill
    annotate("rect", xmin = ox - orf_w, xmax = ox + orf_w,
             ymin = orf_y1, ymax = orf_y2,
             fill = col_fusion, alpha = a, color = "grey40", linewidth = 0.5) +
    annotate("text", x = ox, y = (orf_y1 + orf_y2) / 2,
             label = paste0("ORF ", i), size = 5,
             alpha = if (i == 3) 1 else 0.5)
}

p <- p +
  annotate("text", x = 7.5, y = 16.15,
           label = "K = 5 ORFs per transcript (shared-weight encoder)",
           size = 5.5, fontface = "italic", color = "grey40")

# Arrow from center ORF pill down to encoder detail
p <- p +
  annotate("segment", x = 7.5, xend = 7.5, y = orf_y1, yend = 14.3,
           arrow = arr(), color = col_arrow, linewidth = 0.8) +
  annotate("text", x = 10.0, y = 14.55,
           label = "Detail for one ORF\n(same encoder for all 5)",
           size = 4.5, color = "grey50", fontface = "italic",
           lineheight = 0.85, hjust = 0)

# =========================================================================
# LEVEL 1: Per-ORF Encoder with Visual Inputs (y = 10.0-14.0)
# =========================================================================

# --- Visual sequence strip for ATG branch (x centered at 2.5) ---
strip_x1 <- 1.0; strip_x2 <- 4.0
strip_base <- 13.0
ch_h <- 0.11  # height per channel row
ch_names <- c("A", "C", "G", "T", "Jnc", "GC", "F0", "F1", "F2")

for (i in 1:9) {
  sy1 <- strip_base + (i - 1) * ch_h
  sy2 <- sy1 + ch_h * 0.9
  p <- p +
    annotate("rect", xmin = strip_x1, xmax = strip_x2,
             ymin = sy1, ymax = sy2,
             fill = ch_cols[i], alpha = 0.7, color = NA)
}
# Border around strip
p <- p +
  annotate("rect", xmin = strip_x1, xmax = strip_x2,
           ymin = strip_base, ymax = strip_base + 9 * ch_h,
           fill = NA, color = "grey40", linewidth = 0.4) +
  annotate("text", x = 2.5, y = strip_base - 0.2,
           label = "9ch x 500 positions", size = 4.5, color = "grey40") +
  annotate("text", x = 2.5, y = strip_base + 9 * ch_h + 0.2,
           label = "ATG window", size = 5.5, fontface = "bold", color = "grey40")

# Channel labels (left of ATG strip)
for (i in 1:9) {
  sy <- strip_base + (i - 0.5) * ch_h
  p <- p +
    annotate("text", x = strip_x1 - 0.15, y = sy, label = ch_names[i],
             size = 3.2, hjust = 1, color = "grey30")
}

# --- Visual sequence strip for Stop branch (x centered at 7.0) ---
strip2_x1 <- 5.5; strip2_x2 <- 8.5

for (i in 1:9) {
  sy1 <- strip_base + (i - 1) * ch_h
  sy2 <- sy1 + ch_h * 0.9
  p <- p +
    annotate("rect", xmin = strip2_x1, xmax = strip2_x2,
             ymin = sy1, ymax = sy2,
             fill = ch_cols[i], alpha = 0.7, color = NA)
}
p <- p +
  annotate("rect", xmin = strip2_x1, xmax = strip2_x2,
           ymin = strip_base, ymax = strip_base + 9 * ch_h,
           fill = NA, color = "grey40", linewidth = 0.4) +
  annotate("text", x = 7.0, y = strip_base - 0.2,
           label = "9ch x 500 positions", size = 4.5, color = "grey40") +
  annotate("text", x = 7.0, y = strip_base + 9 * ch_h + 0.2,
           label = "Stop window", size = 5.5, fontface = "bold", color = "grey40")

# --- Visual feature table for Structural branch (x centered at 11.5) ---
feat_x1 <- 10.0; feat_x2 <- 13.0
feat_base <- 13.0
feat_h <- 0.18
feat_names <- c("frac_start", "frac_stop", "is_ref_cds", "is_sqanti_cds", "n_downstream_ejc")
feat_alphas <- c(0.4, 0.5, 0.7, 0.6, 0.9)

for (i in 1:5) {
  fy1 <- feat_base + (i - 1) * feat_h
  fy2 <- fy1 + feat_h * 0.9
  p <- p +
    annotate("rect", xmin = feat_x1, xmax = feat_x2,
             ymin = fy1, ymax = fy2,
             fill = col_struct, alpha = feat_alphas[i],
             color = "grey50", linewidth = 0.3) +
    annotate("text", x = feat_x2 + 0.15, y = (fy1 + fy2) / 2,
             label = feat_names[i], size = 3.5, hjust = 0, color = "grey30")
}
p <- p +
  annotate("text", x = 11.5, y = feat_base - 0.2,
           label = "5 features", size = 4.5, color = "grey40") +
  annotate("text", x = 11.5, y = feat_base + 5 * feat_h + 0.15,
           label = "Structural", size = 5.5, fontface = "bold", color = "grey40")

# --- Arrows from visual inputs to encoder boxes ---
enc_y2 <- 12.5; enc_y1 <- 10.8
p <- p +
  annotate("segment", x = 2.5, xend = 2.5, y = strip_base, yend = enc_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +
  annotate("segment", x = 7.0, xend = 7.0, y = strip_base, yend = enc_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +
  annotate("segment", x = 11.5, xend = 11.5, y = feat_base, yend = enc_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7)

# --- Encoder boxes ---
p <- p +
  # ATG CNN
  annotate("rect", xmin = 0.8, xmax = 4.2, ymin = enc_y1, ymax = enc_y2,
           fill = col_atg, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 2.5, y = 12.0, label = "ATG CNN",
           size = 7, fontface = "bold") +
  annotate("text", x = 2.5, y = 11.5, label = "Conv1D x2 + Pool",
           size = 5, color = "grey30") +
  annotate("text", x = 2.5, y = 11.1, label = paste0("\u2192", " 32-dim"),
           size = 5, color = "grey30") +

  # Stop CNN
  annotate("rect", xmin = 5.3, xmax = 8.7, ymin = enc_y1, ymax = enc_y2,
           fill = col_stop, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 7.0, y = 12.0, label = "Stop CNN",
           size = 7, fontface = "bold") +
  annotate("text", x = 7.0, y = 11.5, label = "Conv1D x2 + Pool",
           size = 5, color = "grey30") +
  annotate("text", x = 7.0, y = 11.1, label = paste0("\u2192", " 32-dim"),
           size = 5, color = "grey30") +

  # Structural Linear
  annotate("rect", xmin = 9.8, xmax = 13.2, ymin = enc_y1, ymax = enc_y2,
           fill = col_struct, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 11.5, y = 12.0, label = "Structural",
           size = 7, fontface = "bold") +
  annotate("text", x = 11.5, y = 11.5, label = "Linear + ReLU",
           size = 5, color = "grey30") +
  annotate("text", x = 11.5, y = 11.1, label = paste0("\u2192", " 32-dim"),
           size = 5, color = "grey30")

# Level label
p <- p +
  annotate("text", x = -2.5, y = 11.65, label = "1", size = 8,
           fontface = "bold", color = "grey60")

# =========================================================================
# FUSION (y = 9.0-10.3)
# =========================================================================
fus_y1 <- 9.0; fus_y2 <- 10.3

p <- p +
  # Converging arrows from three encoder boxes
  annotate("segment", x = 2.5, xend = 7.0, y = enc_y1, yend = fus_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +
  annotate("segment", x = 7.0, xend = 7.0, y = enc_y1, yend = fus_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +
  annotate("segment", x = 11.5, xend = 7.0, y = enc_y1, yend = fus_y2,
           arrow = arr(0.25), color = col_arrow, linewidth = 0.7) +

  # Fusion box
  annotate("rect", xmin = 4.5, xmax = 9.5, ymin = fus_y1, ymax = fus_y2,
           fill = col_fusion, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 7.0, y = 9.85, label = "Concatenate + Fusion",
           size = 7, fontface = "bold") +
  annotate("text", x = 7.0, y = 9.35,
           label = paste0("96-dim ", "\u2192", " 64-dim ORF embedding"),
           size = 5.5, color = "grey30")

# =========================================================================
# 5 ORF EMBEDDINGS → ATTENTION (y = 6.0-8.5)
# =========================================================================

# 5 embedding rectangles
emb_xs <- c(3, 5, 7, 9, 11)
emb_w <- 0.7
emb_y1 <- 8.0; emb_y2 <- 8.5

# Arrow from fusion to embedding row
p <- p +
  annotate("segment", x = 7.0, xend = 7.0, y = fus_y1, yend = emb_y2 + 0.15,
           color = col_arrow, linewidth = 0.7)

# Spread bar and embedding pills
p <- p +
  annotate("segment", x = emb_xs[1] - emb_w, xend = emb_xs[5] + emb_w,
           y = emb_y2 + 0.15, yend = emb_y2 + 0.15,
           color = "grey50", linewidth = 0.6)

for (i in seq_along(emb_xs)) {
  ex <- emb_xs[i]
  p <- p +
    annotate("segment", x = ex, xend = ex,
             y = emb_y2 + 0.15, yend = emb_y2,
             color = "grey50", linewidth = 0.6) +
    annotate("rect", xmin = ex - emb_w, xmax = ex + emb_w,
             ymin = emb_y1, ymax = emb_y2,
             fill = col_fusion, alpha = 0.5, color = "grey40", linewidth = 0.5) +
    annotate("text", x = ex, y = (emb_y1 + emb_y2) / 2,
             label = paste0("e", i), size = 5, fontface = "bold") +
    # Arrow from each embedding to attention box
    annotate("segment", x = ex, xend = 7.0, y = emb_y1, yend = 7.5,
             arrow = arr(0.2), color = col_arrow, linewidth = 0.5)
}

# Attention box
att_y1 <- 6.0; att_y2 <- 7.5
p <- p +
  annotate("rect", xmin = 4.0, xmax = 10.0, ymin = att_y1, ymax = att_y2,
           fill = col_attn, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 7.0, y = 7.1, label = "Attention Aggregator",
           size = 7, fontface = "bold") +
  annotate("text", x = 7.0, y = 6.65,
           label = "Softmax weights across K ORFs",
           size = 5.5, color = "grey30") +
  annotate("text", x = 7.0, y = 6.25,
           label = paste0("\u2192", " 64-dim transcript embedding"),
           size = 5.5, color = "grey30")

# Level label
p <- p +
  annotate("text", x = -2.5, y = (att_y1 + att_y2) / 2, label = "2", size = 8,
           fontface = "bold", color = "grey60")

# =========================================================================
# LEVEL 3: Classification Head (y = 4.0-5.5)
# =========================================================================
cls_y1 <- 4.0; cls_y2 <- 5.5

p <- p +
  annotate("segment", x = 7.0, xend = 7.0, y = att_y1, yend = cls_y2,
           arrow = arr(), color = col_arrow, linewidth = 0.8) +
  annotate("rect", xmin = 4.5, xmax = 9.5, ymin = cls_y1, ymax = cls_y2,
           fill = col_head, alpha = 0.7, color = "grey30", linewidth = 0.8) +
  annotate("text", x = 7.0, y = 5.0, label = "Classification Head",
           size = 7, fontface = "bold") +
  annotate("text", x = 7.0, y = 4.5,
           label = paste0("64 ", "\u2192", " 32 ", "\u2192", " 1 (NMD logit)"),
           size = 5.5, color = "grey30")

# Level label
p <- p +
  annotate("text", x = -2.5, y = (cls_y1 + cls_y2) / 2, label = "3", size = 8,
           fontface = "bold", color = "grey60")

# =========================================================================
# INTERPRETATION: Joint DeepSHAP (y = -0.5 to 3.5)
# =========================================================================
shap_y1 <- -0.3; shap_y2 <- 3.5

p <- p +
  annotate("rect", xmin = 0.0, xmax = 14.0, ymin = shap_y1, ymax = shap_y2,
           fill = col_shap, alpha = 0.2, color = col_jds, linewidth = 1.2,
           linetype = "dashed") +
  annotate("text", x = 7.0, y = 3.1,
           label = "Joint DeepSHAP Interpretation",
           size = 7, fontface = "bold", color = col_jds) +
  annotate("text", x = 7.0, y = 2.6,
           label = "All rank-0 ORF inputs varied simultaneously (9,005 dimensions)",
           size = 5.5, color = col_jds) +
  annotate("text", x = 7.0, y = 2.15,
           label = "500 background samples \u00B7 5 replicates \u00B7 All 15,574 test transcripts",
           size = 4.5, color = "grey40")

# Three SHAP component boxes
p <- p +
  # ATG SHAP
  annotate("rect", xmin = 0.5, xmax = 4.5, ymin = 0.3, ymax = 1.8,
           fill = col_atg, alpha = 0.5, color = "grey40", linewidth = 0.5) +
  annotate("text", x = 2.5, y = 1.35, label = "ATG SHAP",
           size = 6, fontface = "bold") +
  annotate("text", x = 2.5, y = 0.75,
           label = "4,500 values\n(9 \u00D7 500 positions)",
           size = 4.5, color = "grey30", lineheight = 0.85) +

  # Stop SHAP
  annotate("rect", xmin = 5.0, xmax = 9.0, ymin = 0.3, ymax = 1.8,
           fill = col_stop, alpha = 0.5, color = "grey40", linewidth = 0.5) +
  annotate("text", x = 7.0, y = 1.35, label = "Stop SHAP",
           size = 6, fontface = "bold") +
  annotate("text", x = 7.0, y = 0.75,
           label = "4,500 values\n(9 \u00D7 500 positions)",
           size = 4.5, color = "grey30", lineheight = 0.85) +

  # Structural SHAP
  annotate("rect", xmin = 9.5, xmax = 13.5, ymin = 0.3, ymax = 1.8,
           fill = col_struct, alpha = 0.5, color = "grey40", linewidth = 0.5) +
  annotate("text", x = 11.5, y = 1.35, label = "Structural SHAP",
           size = 6, fontface = "bold") +
  annotate("text", x = 11.5, y = 0.75, label = "5 values",
           size = 4.5, color = "grey30")

# Additivity note
p <- p +
  annotate("text", x = 7.0, y = -0.1,
           label = "Additive:  \u03A3 SHAP(ATG) + \u03A3 SHAP(Stop) + \u03A3 SHAP(Struct) = f(x) \u2212 E[f(x)]",
           size = 5.5, color = col_jds, fontface = "italic")

# =========================================================================
# SHAP METHOD ANNOTATIONS (right side brackets)
# =========================================================================
bx <- 15.0  # bracket x position
tick_l <- 0.5  # tick length

# Joint DeepSHAP bracket (encoder inputs, y = 10.8 to 14.2)
jds_y1 <- enc_y1; jds_y2 <- strip_base + 9 * ch_h + 0.4
p <- p +
  annotate("segment", x = bx, xend = bx, y = jds_y1, yend = jds_y2,
           color = col_jds, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = jds_y1, yend = jds_y1,
           color = col_jds, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = jds_y2, yend = jds_y2,
           color = col_jds, linewidth = 1.5) +
  annotate("text", x = bx + 0.3, y = (jds_y1 + jds_y2) / 2,
           label = "Joint\nDeepSHAP", size = 5, fontface = "bold",
           color = col_jds, hjust = 0, lineheight = 0.85)

# KernelSHAP bracket (fusion layer, y = 9.0 to 10.3)
p <- p +
  annotate("segment", x = bx, xend = bx, y = fus_y1, yend = fus_y2,
           color = col_ks, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = fus_y1, yend = fus_y1,
           color = col_ks, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = fus_y2, yend = fus_y2,
           color = col_ks, linewidth = 1.5) +
  annotate("text", x = bx + 0.3, y = (fus_y1 + fus_y2) / 2,
           label = "KernelSHAP\n(branches)", size = 5, fontface = "bold",
           color = col_ks, hjust = 0, lineheight = 0.85)

# Attention weights bracket (aggregator, y = 6.0 to 7.5)
p <- p +
  annotate("segment", x = bx, xend = bx, y = att_y1, yend = att_y2,
           color = col_att, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = att_y1, yend = att_y1,
           color = col_att, linewidth = 1.5) +
  annotate("segment", x = bx - tick_l, xend = bx, y = att_y2, yend = att_y2,
           color = col_att, linewidth = 1.5) +
  annotate("text", x = bx + 0.3, y = (att_y1 + att_y2) / 2,
           label = "Attention\nweights", size = 5, fontface = "bold",
           color = col_att, hjust = 0, lineheight = 0.85)

cat("Architecture figure built successfully\n")
