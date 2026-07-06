import os
import re
import pymupdf

from config import PDF_FOLDER, FIGURE_IMAGE_FOLDER


def find_figure_caption(page_text: str):
    """
    Finds captions like:
    Figure 1: Transformer architecture
    Fig. 2. Model overview
    FIGURE 3 Experimental setup
    """

    patterns = [
        r"(Figure\s+\d+[:.\-\s].{0,300})",
        r"(Fig\.\s+\d+[:.\-\s].{0,300})",
        r"(FIGURE\s+\d+[:.\-\s].{0,300})",
    ]

    for pattern in patterns:
        match = re.search(pattern, page_text, re.IGNORECASE | re.DOTALL)
        if match:
            caption = match.group(1)
            caption = " ".join(caption.split())
            return caption

    return "No figure caption found on this page."


def extract_figures_from_pdfs():
    os.makedirs(FIGURE_IMAGE_FOLDER, exist_ok=True)

    extracted_items = []

    for pdf_file in os.listdir(PDF_FOLDER):
        if not pdf_file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]

        print(f"Processing figures from: {pdf_file}")

        doc = pymupdf.open(pdf_path)

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            page_text = page.get_text() or ""

            caption = find_figure_caption(page_text)

            images = page.get_images(full=True)

            for image_index, img in enumerate(images):
                xref = img[0]

                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    image_name = (
                        f"{pdf_name}_page_{page_number}"
                        f"_figure_{image_index + 1}.{image_ext}"
                    )

                    image_path = os.path.join(FIGURE_IMAGE_FOLDER, image_name)

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    extracted_items.append(
                        {
                            "paper_name": pdf_file,
                            "page_number": page_number,
                            "figure_index": image_index + 1,
                            "image_path": image_path,
                            "caption": caption,
                        }
                    )

                    print(f"Saved: {image_path}")

                except Exception as e:
                    print(f"Skipping image on page {page_number}: {e}")

        doc.close()

    return extracted_items


if __name__ == "__main__":
    figures = extract_figures_from_pdfs()
    print(f"Total figures extracted: {len(figures)}")