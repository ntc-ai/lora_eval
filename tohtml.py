import os
import shutil
import collections
import math

def parse_float_from_filename(filename):
    # Extract the float value from the filename
    try:
        return float(filename.rsplit('.', 1)[0].rsplit('_', 1)[-1])
    except ValueError:
        return 0.0


def get_file_ctime(subdir):
  # Extract the filename from the subdir tuple
  filename = subdir[1] + ".safetensors"
  # Get the full path to the file
  full_path = os.path.join('loras', filename)
  # Check if the file exists
  if os.path.exists(full_path):
    # Get the creation time of the file
    return os.path.getctime(full_path)
  else:
    # Raise an error if the file doesn't exist
    raise FileNotFoundError(f"File {full_path} does not exist")

def scan_directory(base_path):
    images = collections.defaultdict(lambda: collections.defaultdict(list))
    subdirs = [(os.path.join(base_path, subdir), subdir) for subdir in next(os.walk(base_path))[1]]
    subdirs = [t for t in subdirs if 'loras' != t[1] and 'body contorted' not in t[1]]
    subdirs.sort(key=get_file_ctime, reverse=True)

    for subdir_path, subdir in subdirs:
        for file in os.listdir(subdir_path):
            if file.endswith(".png"):
                parts = file.split('_')
                word_seed = f'{parts[0]}_{parts[1]}'
                full_path = os.path.join(subdir, file)
                images[subdir][word_seed].append(full_path)

        for word_seed in images[subdir].keys():
            images[subdir][word_seed].sort(key=parse_float_from_filename)

    return images, [subdir for _, subdir in subdirs]

def generate_subdir_html(subdir, images):
    modalcss = modal_style()
    html_content = f'<!DOCTYPE html>\n<html>\n<head>\n<title>{subdir}</title>\n<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">\n{modalcss}</head>\n<body>\n<div class="mx-auto px-4">\n'
    name = " -> ".join(reversed(subdir.replace("..", ".").split(".")))
    html_content += f'<h2 class="text-2xl font-bold my-4">{name}</h2>\n'
    html_content += f'<div class="mb-6"><a class="bp-6 px-6 py-3 text-lg text-white font-bold bg-blue-600 hover:bg-blue-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-300 focus:ring-opacity-50 shadow-md transition duration-300 ease-in-out" href="/loras/{subdir}.safetensors">Download</a></div>\n'

    for word_seed, files in images.items():
        html_content += f'<div class="flex flex-row flex-wrap justify-center items-center mb-4">\n'
        for file in files:
            html_content += f'  <img src="{file}" class="img-modal max-w-full h-auto md:w-1/5 lg:w-1/5 xl:w-1/5" alt="{file}"/>\n'
        html_content += '</div>\n'

    html_content += '</div>\n{modal_script}</body>\n</html>'.format(modal_script=modal_script())
    return html_content


def generate_pagination_links(current_page, total_pages):
    links = '<div class="flex justify-between items-center my-4 text-xl">\n'
    if current_page > 1:
        links += f'<a href="index_{current_page - 1}.html" class="flex items-center"><span>&#9664;</span> Previous</a>\n'
    else:
        links += '<span></span>\n'  # Empty span for spacing

    links += '<div>'
    for i in range(1, total_pages + 1):
        if i == current_page:
            links += f'<span class="current-page mx-1">{i}</span>\n'
        else:
            links += f'<a href="index_{i}.html" class="text-blue-500 hover:text-blue-700 mx-1">{i}</a>\n'
    links += '</div>'

    if current_page < total_pages:
        links += f'<a href="index_{current_page + 1}.html" class="flex items-center">Next <span>&#9654;</span></a>\n'
    else:
        links += '<span></span>\n'  # Empty span for spacing
    links += '</div>\n'
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

def generate_index_html(subdirs, images, page, total_pages):

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
    html_content += '<h1 class="text-3xl font-bold my-4">Loras (freshly updated)</h1>\n'
    html_content += generate_pagination_links(page, total_pages)
    html_content += '<ul>\n'
    start = (page - 1) * 8
    end = start + 8
    displayed_subdirs = subdirs[start:end]


    for subdir in displayed_subdirs:
        word_seeds = list(images[subdir].keys())
        first_word_seed = word_seeds[0] if word_seeds else None

        html_content += f'<li>\n'
        name = " -> ".join(reversed(subdir.replace("..", ".").split(".")))
        html_content += f'<a href="{subdir}.html" class="text-blue-500 hover:text-blue-700 text-xl">{name}</a>\n'

        if first_word_seed:
            html_content += f'<div class="flex flex-row flex-wrap justify-center items-center mb-4">\n'
            for file in images[subdir][first_word_seed][:5]:  # Adjust the number of images as needed
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

def main():
    base_directory_path = 'evaluate' # Set this to the path of your base directory
    images, subdirs = scan_directory(base_directory_path)

    # Generate individual HTML files for each subdirectory
    for subdir in subdirs:
        subdir_html = generate_subdir_html(subdir, images[subdir])
        with open(f'{base_directory_path}/{subdir}.html', 'w') as file:
            file.write(subdir_html)

    total_pages = math.ceil(len(subdirs) / 8)
    for page in range(1, total_pages + 1):
        index_html = generate_index_html(subdirs, images, page, total_pages)
        with open(f'{base_directory_path}/index_{page}.html', 'w') as file:
            file.write(index_html)

    # Rename first index page for ease of access
    shutil.copyfile(f'{base_directory_path}/index_1.html', f'{base_directory_path}/index.html')

if __name__ == "__main__":
    main()

