import os
import json
import shutil
import collections
import math
import argparse

def get_file_ctime(subdir):
  # Extract the filename from the subdir tuple
  #filename = subdir[1] + ".safetensors"
  # Get the full path to the file
  #full_path = os.path.join('loras', filename)
  # Check if the file exists
  #if os.path.exists(full_path):
    # Get the creation time of the file
  return os.path.getctime(subdir[0])



def scan_directory(base_path):
    images = collections.defaultdict(lambda: collections.defaultdict(list))
    subdirs = [(os.path.join(base_path, subdir), subdir) for subdir in next(os.walk(base_path))[1]]
    subdirs = [t for t in subdirs if 'loras' not in t[1]]
    subdirs.sort(key=get_file_ctime, reverse=True)

    for subdir_path, subdir in subdirs:
        for file in os.listdir(subdir_path):
            if file.endswith(".png"):
                word_seed="1"
                full_path = os.path.join(subdir, file)
                images[subdir][word_seed].append(full_path)


    return images, [subdir for _, subdir in subdirs]

def generate_subdir_html(subdir, images, base_dir):
    data = open(base_dir+"/"+subdir+".json", "r").read()
    modalcss = modal_style()
    html_content = f'<!DOCTYPE html>\n<html>\n<head>\n<title>{subdir}</title>\n<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">\n{modalcss}</head>\n<body>\n<div class="mx-auto px-4">\n'
    #name = " -> ".join(reversed(subdir.replace("..", ".").split(".")))
    name = subdir
    prompt = ', '.join([phrase.split('.')[0] for phrase in json.loads(data)['loras']])
    html_content += f'<h2 class="text-2xl font-bold my-4">{name}</h2>\n'
    html_content += "<pre><code>"+data+"</code></pre>"
    html_content += f"<p>Instructions:<pre><code>Prompt with\n\n{prompt}\n\n</code></pre></p>"
    html_content += f'<div class="mb-6"><a class="bp-6 px-6 py-3 text-lg text-white font-bold bg-blue-600 hover:bg-blue-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-300 focus:ring-opacity-50 shadow-md transition duration-300 ease-in-out" href="/combo_loras/{subdir}_00001_.safetensors">Download</a></div>\n'

    def get_image_ctime(subdir):
        img = base_dir+"/"+subdir
        return os.path.getctime(img)
    for word_seed, files in images.items():
        html_content += f'<div class="flex flex-row flex-wrap justify-center items-center mb-4">\n'
        imgs = files
        imgs.sort(key=get_image_ctime)
        html_content += "Random samples at 1</div>"
        html_content += f'<div class="flex flex-row flex-wrap justify-center items-center mb-4">\n'
        for file in imgs:  # Adjust the number of images as needed
            html_content += f'  <img src="{file}" class="img-modal h-auto md:w-1/5 lg:w-1/5 xl:w-1/5" alt="{file}"/>\n'
        html_content += "</div>"

    html_content += '</div>\n{modal_script}</body>\n</html>'.format(modal_script=modal_script())
    return html_content

def generate_pagination_links(current_page, total_pages):
    links = '<nav aria-label="Page navigation" class="flex justify-between items-center my-4 text-xl">\n'

    # Link to the previous page
    if current_page > 1:
        links += f'<a href="index_{current_page - 1}.html" class="hover:text-blue-600">Previous Page</a>\n'
    else:
        links += '<span></span>\n'  # Empty span for spacing

    # Page navigation links
    links += '<div class="flex items-center">'

    # Always show the first page
    if current_page != 1:
        links += '<a href="index_1.html" class="mx-1">Page 1</a>\n'
    else:
        links += '<a href="index_1.html" class="mx-1">Page</a>\n'

    # Add ellipsis if there are pages between first page and current page
    if current_page > 3:
        links += '<span class="mx-1">...</span>\n'

    # Show up to 5 page links leading up to the current page
    page_range_start = max(2, current_page - 2)
    for i in range(page_range_start, current_page):
        links += f'<a href="index_{i}.html" class="mx-1">{i}</a>\n'

    # Show the current page
    links += f'<span class="current-page mx-1 bg-blue-500 text-white px-2 py-1 rounded-md">{current_page}</span>\n'

    # Show up to 5 page links after the current page
    page_range_end = min(current_page + 3, total_pages)
    for i in range(current_page + 1, page_range_end):
        links += f'<a href="index_{i}.html" class="mx-1">{i}</a>\n'

    # Add ellipsis if there are more pages after the range
    if current_page < total_pages - 3:
        links += '<span class="mx-1">...</span>\n'

    # Always show the last page
    if current_page < total_pages:
        links += f'<a href="index_{total_pages}.html" class="mx-1">{total_pages}</a>\n'

    links += '</div>\n'

    # Link to the next page
    if current_page < total_pages:
        links += f'<a href="index_{current_page + 1}.html" class="hover:text-blue-600">Next Page</a>\n'
    else:
        links += '<span></span>\n'  # Empty span for spacing

    links += '</nav>\n'
    return links

def modal_style():
    return """<style>
            .modal { display: none; position: fixed; z-index: 10; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.8); }
            .modal-content { margin: 2% auto; padding: 20px; }
            .close { color: #aaa; float: right; font-size: 28px; font-weight: bold; }
            .close:hover, .close:focus { color: black; text-decoration: none; cursor: pointer; }
            img { cursor: pointer; }
        </style>"""

def modal_script():
    return """<!-- The Modal -->
        <div id="myModal" class="modal">
            <span class="close">&times;</span>
            <img class="modal-content" id="img01">
        </div>

        <script>
            // Get the modal
            var modal = document.getElementById('myModal');
            window.onclick = function(event) {{
                if (event.target == modal) {{
                    modal.style.display = 'none';
                }}
            }};


            // Get the image and insert it inside the modal
            var modalImg = document.getElementById('img01');
            var closeBtn = document.getElementsByClassName('close')[0];

            document.querySelectorAll('.img-modal').forEach(item => {{
                item.onclick = function() {{
                    modal.style.display = 'block';
                    modalImg.src = this.src;
                }}
            }});

            // When the user clicks on <span> (x), close the modal
            closeBtn.onclick = function() {{
                modal.style.display = 'none';
            }}
        </script>"""

def generate_index_html(subdirs, images, page, total_pages, per_page, base_dir):

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Directory Index</title>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">

        {modal_style}
    </head>
    <body>
        <div class="container mx-auto px-4">
    """.format(modal_style=modal_style())
    html_content += '<h1 class="text-3xl font-bold my-4">Combined Loras (live feed)</h1>\n'
    html_content += generate_pagination_links(page, total_pages)
    html_content += '<ul>\n'
    start = (page - 1) * per_page
    end = start + per_page
    displayed_subdirs = subdirs[start:end]

    def get_image_ctime(subdir):
        img = base_dir+"/"+subdir
        return os.path.getctime(img)

    for subdir in displayed_subdirs:
        data = open(base_dir+"/"+subdir+".json", "r").read()
        length = len(json.loads(data)["loras"])
        modalcss = modal_style()
        word_seeds = list(images[subdir].keys())
        first_word_seed = word_seeds[0] if word_seeds else None

        if first_word_seed is None or len(images[subdir][first_word_seed]) < 5:
            continue

        html_content += f'<li>\n'
        #name = " -> ".join(reversed(subdir.replace("..", ".").split(".")))
        name = subdir
        html_content += f'<a href="{subdir}.html" class="text-blue-500 hover:text-blue-700 text-xl">{name}</a> (<b>{length}</b> loras merged)\n'

        if first_word_seed:
            html_content += f'<div class="flex flex-row flex-wrap justify-center items-center mb-4">\n'
            imgs = images[subdir][first_word_seed][:15]
            imgs.sort(key=get_image_ctime)
            for file in imgs[-5:]:  # Adjust the number of images as needed
                html_content += f'  <img src="{file}" class="img-modal h-auto md:w-1/5 lg:w-1/5 xl:w-1/5" alt="{file}"/>\n'
            html_content += '</div>\n'
        html_content += '</li>\n'


    html_content += """
            </ul>
            {pagination_links}
        </div>

        {modal_script}
    </body>
    </html>""".format(pagination_links=generate_pagination_links(page, total_pages), modal_script=modal_script())

    return html_content

    parser.add_argument('-pval', '--p_value', type=float, default=0.25, help="between 0 and 1")
def main():
    parser = argparse.ArgumentParser(description='Generate images for evaluating a lora')
    parser.add_argument('-b', '--base', type=str, default="combo_eval", help='directory to read from')
    parser.add_argument('-p', '--per_page', type=int, default=8, help='items per page')
    args = parser.parse_args()

    images, subdirs = scan_directory(args.base)

    # Generate individual HTML files for each subdirectory
    for subdir in subdirs:
        subdir_html = generate_subdir_html(subdir, images[subdir], args.base)
        with open(f'{args.base}/{subdir}.html', 'w') as file:
            file.write(subdir_html)

    total_pages = math.ceil(len(subdirs) / args.per_page)
    for page in range(1, total_pages + 1):
        index_html = generate_index_html(subdirs, images, page, total_pages, args.per_page, args.base)
        with open(f'{args.base}/index_{page}.html', 'w') as file:
            file.write(index_html)

    # Rename first index page for ease of access
    shutil.copyfile(f'{args.base}/index_1.html', f'{args.base}/index.html')

if __name__ == "__main__":
    main()

