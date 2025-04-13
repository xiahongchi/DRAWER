import requests
import base64
import os
import json
from PIL import Image
import time
import argparse
import subprocess

class GPT:
    """
    Simple interface for interacting with GPT-4O model
    """
    VERSIONS = {
        "4v": "gpt-4-vision-preview",
        "4o": "gpt-4o",
        "4o-mini": "gpt-4o-mini",
    }

    def __init__(
            self,
            api_key,
            version="4o",
    ):
        """
        Args:
            api_key (str): Key to use for querying GPT
            version (str): GPT version to use. Valid options are: {4o, 4o-mini, 4v}
        """
        self.api_key = api_key
        assert version in self.VERSIONS, f"Got invalid GPT version! Valid options are: {self.VERSIONS}, got: {version}"
        self.version = version

    def __call__(self, payload, verbose=False):
        """
        Queries GPT using the desired @prompt

        Args:
            payload (dict): Prompt payload to pass to GPT. This should be formatted properly, see
                https://platform.openai.com/docs/overview for details
            verbose (bool): Whether to be verbose as GPT is being queried

        Returns:
            None or str: Raw outputted GPT response if valid, else None
        """
        if verbose:
            print(f"Querying GPT-{self.version} API...")

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.query_header, json=payload)
        if "choices" not in response.json().keys():
            print(f"Got error while querying GPT-{self.version} API! Response:\n\n{response.json()}")
            return None

        if verbose:
            print(f"Finished querying GPT-{self.version}.")

        return response.json()["choices"][0]["message"]["content"]

    
    @property
    def query_header(self):
        """
        Returns:
            dict: Relevant header to pass to all GPT queries
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def encode_image(self, image_path):
        """
        Encodes image located at @image_path so that it can be included as part of GPT prompts

        Args:
            image_path (str): Absolute path to image to encode

        Returns:
            str: Encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    

    def payload_get_drawer_recognition(self, img_path):
        """
        Generates custom prompt payload for object captioning

        Args:
            img_path (str): Absolute path to image to caption

        Returns:
            dict: Prompt payload
        """
        # Getting the base64 string
        base64_image = self.encode_image(img_path)

        prompting_text_system = """
```
### Task Description ###
The user will provide an image of a scene with a target object highlighted by a green mask. Blue bounding boxes in the image indicate objects of interest (typically handles).

Your task is to determine if the green-masked object is a VALID SINGLE OPENABLE DOOR/DRAWER belonging to FURNITURE or BUILT-IN CABINETRY by evaluating five criteria.

### CRITICAL RULE - APPLIANCE DETECTION ###
- IMPORTANT: If the green mask covers ANY FULL APPLIANCE (refrigerator, dishwasher, oven, etc.), output `N` immediately.
- IMPORTANT: Refrigerator identification: If the object is tall (floor to counter/ceiling), colored differently than surrounding cabinetry, and has a handle, it is almost certainly a full refrigerator - output `N`.
- When in doubt about whether an object is a full appliance vs. just a door, output `N`.

### Critical Decision Steps ###

1. **Identify Object Category:**
   - VALID categories (furniture/cabinetry openable elements ONLY):
     - Kitchen cabinet door
     - Bathroom cabinet door
     - Closet/wardrobe door
     - Built-in shelving door
     - Desk drawer/door
     - Dresser drawer/door
     - Drawer front/face
     - Individual door of a home appliance (NOT the entire appliance)
   - WHOLE APPLIANCES (ALWAYS INVALID - output `N`):
     - Full refrigerators (even single-door models)
     - Full dishwashers
     - Full ovens
     - Full microwaves
     - Full washing machines
     - Full dryers
   - OTHER INVALID categories (not furniture/cabinetry):
     - Office equipment panels
     - Electronic device doors/covers
     - Electrical panels/boxes
     - Industrial equipment access doors
     - Vehicle compartments/doors
     - Coffee machine/filter related surfaces
   - If the object doesn't match a VALID category OR is a WHOLE APPLIANCE, output: `N`

2. **Size and Scale Assessment:**
   - CRITICAL: Examine the size and dimensions of the masked object
   - If the masked object extends from floor to counter/ceiling height and appears to be a full-sized appliance like a refrigerator, output: `N`
   - If the masked object has different coloring or finishing from surrounding cabinetry, it's likely an appliance, not a cabinet door
   - If the masked object appears to be the full height and width of a standard appliance, output: `N`
   - Key indicators of whole refrigerators: tall height, distinctive color/material different from surrounding cabinetry, rounded edges, ventilation at top/bottom

3. **Handle and Opening Mechanism Assessment:**
   - Examine handles indicated by blue bounding boxes
   - For cabinet doors: handles are usually centrally placed on the door
   - For refrigerators: a single handle on a tall unit almost always indicates a full refrigerator (output `N`), not just a door
   - CRITICAL: The handle position and type can help identify whether the object is a full appliance (invalid) or just a door (potentially valid)
   - If the handle appears to be a refrigerator-style handle on a large unit, output: `N`

4. **Cabinet Structure Assessment:**
   - Compare the masked object to adjacent cabinet elements
   - If the masked object has a significantly different size, color, or finish from surrounding cabinetry, it's likely an appliance not a cabinet door
   - SPECIAL CASE - REFRIGERATORS: If the object is distinctively colored (different from surrounding cabinets), has a typical refrigerator handle, and extends from floor to counter/ceiling, it is a refrigerator (output `N`)
   - Look for edges, seams, and gaps that distinguish a cabinet door from a full appliance
   
5. **Completeness Check:**
   - A VALID door/drawer must be COMPLETELY covered by the green mask
   - The green mask should contain EXACTLY ONE door/drawer, not multiple
   - If the mask covers what appears to be an entire refrigerator, dishwasher, oven, or other appliance, output: `N`
   - The mask should include the entire operational unit (the complete openable component)
   - If the mask covers multiple doors or only a partial door, output: `N`

### Important Guidelines ###
- WHOLE APPLIANCE RULE: If the masked object appears to be a complete appliance (refrigerator, dishwasher, oven, etc.) rather than just a door of that appliance, output: `N`
- REFRIGERATOR IDENTIFICATION: Even if a refrigerator appears to have only one visible door in the image, if the mask covers the entire refrigerator unit (typically floor to counter/ceiling height, distinctive color/material), output: `N`
- The PRIMARY indicator of a valid door/drawer is the presence of a handle on furniture/cabinetry designed for storage, NOT on a whole appliance
- When in doubt about whether something is a whole appliance versus just a door, classify it as a whole appliance and output: `N`
- Only accept HOUSEHOLD FURNITURE and BUILT-IN CABINETRY storage components, not whole appliances

### Appliance Identification Quick Reference ###
- Refrigerators typically:
  * Extend from floor to counter/ceiling height
  * Have a distinctive color/finish different from surrounding cabinetry
  * Have specialized handles (longer, stainless steel, etc.)
  * Have rounded edges or trim
  * May have a water/ice dispenser
  * May have ventilation grilles at top/bottom
  * RESULT: Always output `N` for full refrigerators

- Dishwashers typically:
  * Are located under countertops
  * Have a control panel along the top edge
  * RESULT: Always output `N` for full dishwashers

- Ovens typically:
  * Have glass door panels
  * Have control panels
  * RESULT: Always output `N` for full ovens

### Output Format ###
Provide your detailed analysis, addressing EACH of the five assessment steps above explicitly.

Your analysis must conclude with EXACTLY ONE of these two possible endings:
- "Final Decision: Y" 
- "Final Decision: N"

Do NOT add any explanatory text after "Final Decision: Y" or "Final Decision: N". The line must end with either "Y" or "N".

### Example Analysis for a Refrigerator ###
1. **Object Category**: The green-masked object appears to be a refrigerator. Its height, coloring (green), and design are consistent with a full refrigerator appliance, not just a single door of cabinetry.

2. **Size and Scale Assessment**: The object extends from floor level to counter/ceiling height, which is typical of a full refrigerator unit. Its size and proportions are characteristic of a complete appliance rather than just a door.

3. **Handle and Opening Mechanism Assessment**: There is a handle (in blue bounding box) on the side of the refrigerator. The handle type and position are consistent with a refrigerator door. However, since this is a full appliance (refrigerator) and not just a door of built-in cabinetry, it fails this criterion.

4. **Cabinet Structure Assessment**: The refrigerator has a distinctive color (green) that differs from the surrounding cabinetry, further indicating it is a separate appliance and not part of the built-in cabinetry. It stands as a complete unit rather than just being a door.

5. **Completeness Check**: The green mask covers the entire refrigerator appliance, not just a single door component. Refrigerators contain multiple compartments (even if only one door is visible in this view), making it an invalid object for our purposes.

Final Decision: N

### Example Analysis for a Valid Cabinet Door ###
1. **Object Category**: The green-masked object appears to be a kitchen cabinet door. It has the same appearance as other cabinet doors in the row.

2. **Purpose and Environment Check**: The object is clearly part of built-in kitchen cabinetry designed for storage in a residential kitchen setting.

3. **Handle Assessment**: There is a handle (in blue bounding box) attached to the green-masked area. The handle is positioned similar to other handles on adjacent doors. The presence of this handle strongly indicates this is an openable door. I observe only ONE handle within the green mask, suggesting this is a single door.

4. **Cabinet Structure Assessment**: The masked panel is part of a row of similar cabinet doors. It has the same finish, dimensions, and style as the adjacent doors that clearly have handles. Although it's at the end of the row, the presence of a handle confirms it's a door, not a side panel. There are no separation lines visible within the green mask that would suggest multiple doors.

5. **Completeness Check**: The green mask completely covers a single cabinet door including its entire operational area. There are no visible seams or divisions within the mask that would indicate multiple doors.

Final Decision: Y

### Example Analysis for Multiple Cabinet Doors ###
1. **Object Category**: The green-masked area appears to contain cabinet doors used for storage in what looks like a kitchen or bathroom environment.

2. **Purpose and Environment Check**: The objects are part of built-in cabinetry designed for storage in what appears to be a residential setting.

3. **Handle Assessment**: I observe TWO handles (in blue bounding boxes) within the green-masked area. Each handle appears to be attached to a separate door panel. The presence of multiple handles indicates multiple doors within the mask.

4. **Cabinet Structure Assessment**: Looking at the structure, I can see a clear separation line running vertically between the two handles, dividing the green-masked area into two distinct doors. Each door has its own handle and appears to be a separate operational unit.

5. **Completeness Check**: While the green mask covers complete doors, it encompasses TWO separate doors rather than a single door. The presence of two handles and the visible separation line between them confirms these are multiple distinct doors.

Final Decision: N

### Example Analysis for Non-Cabinet Panel ###
1. **Object Category**: The green-masked object appears to be an access panel on an office printer/copier or similar electronic equipment. This does not fall into any of the valid furniture/cabinetry categories.

2. **Purpose and Environment Check**: The object is part of office equipment, not household furniture or built-in cabinetry. It appears designed for maintenance access to internal components rather than for storage of items.

3. **Handle Assessment**: There is a handle or pull tab (in blue bounding box) on the green-masked panel. While this indicates it's openable, the object itself doesn't qualify as furniture/cabinetry.

4. **Cabinet Structure Assessment**: The panel is part of electronic equipment rather than a cabinet system. It doesn't share characteristics with furniture storage solutions.

5. **Completeness Check**: While the mask covers a single complete panel, the panel itself is not a valid furniture/cabinetry door or drawer.

Final Decision: N
```
"""

        text_dict_system = {
            "type": "text",
            "text": prompting_text_system
        }
        content_system = [text_dict_system]
        
        content_user = [
            {
                "type": "text",
                "text": "Please analyze the image and determine if the green-masked object is a valid openable door/drawer belonging to furniture or built-in cabinetry, following the assessment steps provided."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]

        object_caption_payload = {
            "model": self.VERSIONS[self.version],
            "messages": [
                {
                    "role": "system",
                    "content": content_system
                },
                {
                    "role": "user",
                    "content": content_user
                }
            ],
            "temperature": 0,
            "max_tokens": 800
        }

        return object_caption_payload

def find_all_occurrences(text, substring):
    indices = []
    index = text.find(substring)
    while index != -1:
        indices.append(index)
        index = text.find(substring, index + 1)
    return indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    gpt = GPT(api_key=args.api_key)
    y_list = []
    n_list = []
    f_list = []
    response_dict = {}

    image_dir = os.path.join(args.data_dir, 'perception', "vis_groups_handle_note/")
    save_dir = os.path.join(args.data_dir, 'perception', "vis_groups_gpt4_api/")
    subprocess.run(f"rm {save_dir}/*", shell=True)
    os.makedirs(save_dir, exist_ok=True)

    for fname in sorted(os.listdir(image_dir)):
        print("gpt query for", fname)
        image_path = os.path.join(image_dir, fname)
        payload_drawer_recognition = gpt.payload_get_drawer_recognition(image_path)

        time.sleep(5)

        response = gpt(payload_drawer_recognition)
        response = str(response)

        print("Response:", response[:200] + "..." if len(response) > 200 else response)
        
        # Find the Final Decision in the response
        try:
            index = find_all_occurrences(response, "Final Decision")[-1]
            final_decision_text = response[index:]
            print(final_decision_text)
            
            index += len("Final Decision")

            response_dict[fname] = response

            if "N" in response[index:index+5]:
                n_list.append(fname)
                print(f"Decision: N - {fname}")
            elif "Y" in response[index:index+5]:
                y_list.append(fname)
                Image.open(image_path).save(os.path.join(save_dir, fname))
                print(f"Decision: Y - {fname}")
            else:
                f_list.append(fname)
                print(f"Decision unclear - {fname}")
        except (IndexError, ValueError) as e:
            print(f"Error parsing response for {fname}: {e}")
            f_list.append(fname)
        
    result_dict = {
        "y_list": y_list,
        "n_list": n_list,
        "f_list": f_list,
    }
    
    # Save all responses and results
    with open(os.path.join(save_dir, "response.json"), 'w') as f:
        json.dump(response_dict, f, indent=4)
    with open(os.path.join(save_dir, "result.json"), 'w') as f:
        json.dump(result_dict, f, indent=4)
    
    # Print summary statistics
    print(f"\nSummary:\nValid doors (Y): {len(y_list)}\nInvalid objects (N): {len(n_list)}\nUnclear cases: {len(f_list)}")