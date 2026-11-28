# History rewrite 2026-11-29 — message-only

## Keputusan
Menghapus baris `Co-authored-by: Claude <noreply@anthropic.com>` dari pesan
commit abae93c dan d565a2a beserta descendant-nya. Trailer tidak mencerminkan
alur kerja aktual (patch diterapkan manual oleh pemilik repo), sehingga
atribusi kontributor GitHub keliru.

## Metode
git filter-repo --message-callback pada mirror clone; hanya pesan commit yang
diubah; push paksa main, bakeoff-step2, fase0-sft.

## Verifikasi
- Tree hash sebelum == sesudah: main & bakeoff-step2 =
  3624c81669febbf831c6c1be5d7322ff064c2340; fase0-sft =
  7c66db022949b5aa1cec50661744503cb500738a.
- Per-commit abae93c->3482d3b dan d565a2a->ca0f14e: tree identik dengan
  salinan pra-rewrite di clone lokal.
- pytest pasca-reset: 249 passed.
- grep trailer co-author claude di semua branch: nol match.

## Peta hash
Lihat history-rewrite-2026-11-29.commit-map (old -> new). Kutipan hash
pra-rewrite pada artefak run tetap sah sebagai catatan historis.

## Pencegahan
Hook commit-msg lokal menolak pesan commit ber-trailer Co-authored-by
yang menyebut claude / noreply@anthropic.com.
