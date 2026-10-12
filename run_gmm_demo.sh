python gmm_demo.py \
  --caching_dir caching/B3_Qwen2_2B_cls \
  --subset_name ImageNet_1K N24News HatefulMemes VOC2007 SUN397 \
  --side qry --K 5 --dedup --cov_type diag --out projection_plots/B3_Qwen2_2B_cls

# python gmm_demo.py \
#   --caching_dir caching/FastVLM-0.5B_base_16_eos_cls \
#   --subset_name ImageNet_1K N24News HatefulMemes VOC2007 SUN397 \
#   --side qry --K 5 --dedup --cov_type diag --out projection_plots/FastVLM-0.5B_base_16_eos_cls

python gmm_demo.py \
  --caching_dir caching/FastVLM-0.5B_cls_struct_sigreg_SW_gmm_d1_sig1_kd10_sw0.1_numt17_tmax5_l24 \
  --subset_name ImageNet_1K N24News HatefulMemes VOC2007 SUN397 \
  --side qry --K 5 --dedup --cov_type diag --out projection_plots/FastVLM-0.5B_cls_struct_sigreg_SW_gmm_d1_sig1_kd10_sw0.1_numt17_tmax5_l24