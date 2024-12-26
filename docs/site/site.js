const cases = [
  { number: '01', file: '01 - lol easy.jpg', title: 'The starting point.', description: 'A clean image gives us a baseline: distinct bars, visible spaces, and enough contrast to measure their widths.', alt: 'Original clean barcode test image' },
  { number: '03', file: '03 - eda ya3am ew3a soba3ak mathazarsh.jpg', title: 'Something in the way.', description: 'A foreign object covers part of the barcode. In the local benchmark, this was the one case that did not match.', alt: 'Barcode test image partially covered by a foreign object' },
  { number: '07', file: '07 - mal7 w felfel.jpg', title: 'A little too much noise.', description: 'Salt-and-pepper noise scatters bright and dark pixels through the image, making the barcode harder to isolate.', alt: 'Barcode test image with salt and pepper noise' },
  { number: '09', file: '09 - e3del el soora ya3ammm.jpg', title: 'A different angle.', description: 'A tilted, partially cut image complicates the job of aligning the bars before measuring their widths.', alt: 'Tilted barcode test image' },
  { number: '10', file: '10 - wen el kontraastttt.jpg', title: 'Fading into the background.', description: 'With little contrast between bars and background, the pattern needs enhancement before it can be read.', alt: 'Low contrast barcode test image' }
];
const buttons = document.querySelectorAll('.sample');
buttons.forEach(button => button.addEventListener('click', () => {
  const sample = cases[Number(button.dataset.index)];
  buttons.forEach(item => { const selected = item === button; item.classList.toggle('active', selected); item.setAttribute('aria-pressed', String(selected)); });
  const img = document.getElementById('sample-image');
  img.src = `recovered/test-cases/${encodeURIComponent(sample.file)}`; img.alt = sample.alt;
  document.getElementById('sample-number').textContent = `CASE ${sample.number} / ORIGINAL IMAGE`;
  document.getElementById('sample-mark').textContent = sample.number;
  document.getElementById('sample-title').textContent = sample.title;
  document.getElementById('sample-description').textContent = sample.description;
}));
