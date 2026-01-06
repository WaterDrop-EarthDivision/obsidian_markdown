---
{"dg-publish":true,"permalink":"/tools/kegg/","dgPassFrontmatter":true,"created":"2026-01-06T20:46:24.298+08:00","updated":"2026-01-06T20:55:30.968+08:00"}
---

## 1.获取KEGG的JSON格式 并保存到指定位置
>hsa  人
>mmu 鼠

```R
kk = gson_KEGG("mmu")  

gson::write.gson(kk,file = "data/son_kegg.json") # 保存到指定位置
```

## 2.获取已下载的JSON格式

```R
kk = gson::read.gson("data/son_kegg.json")
```

## 3.KEGG富集

```R
?gseKEGG
FT_up_egg <- enrichKEGG(gene = FT_up_gene_df$ENTREZID,
                        organism = kk,
                        keyType = "ENTREZID",
                        pvalueCutoff = 0.05,
                        pAdjustMethod = "BH",
                        minGSSize = 10,
                        maxGSSize = 500,
                        qvalueCutoff = 0.2,)  # 人是hsa

FT_up_egg <- setReadable(FT_up_egg,
                   OrgDb = org.Mm.eg.db, # 小鼠改为org.Mm.eg.db  Hs
                   keyType = "ENTREZID")  # 表示输入的基因ID是Entrez基因ID
barplot(FT_up_egg)

p = dotplot(FT_up_egg)
p
ggsave("figure/FTvsFA_up_KEGG.pdf",plot = p,height = 5,width = 4)
```