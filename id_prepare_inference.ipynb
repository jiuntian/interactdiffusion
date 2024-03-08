{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "30b735ef-9839-4174-a180-2b24c28f12b8",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from copy import deepcopy\n",
    "import glob\n",
    "import json\n",
    "import os\n",
    "import pickle\n",
    "import shutil\n",
    "\n",
    "from PIL import Image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "d77c4cfa-de63-4bcc-9971-d077ed4c9252",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "9546\n"
     ]
    }
   ],
   "source": [
    "anno_path = 'DATA/test_hico_ann.json'  # hico-det annotation file\n",
    "output_path = 'generated_output/'  # directory of generated images,\n",
    "new_dataset_path = '../FGAHOI/data/id_generated_output/'  # target FGAHOI-formatted dataset path\n",
    "res_multi = pickle.load(open('DATA/hico_det_test.pkl', 'rb'))\n",
    "annos = json.load(open(anno_path))\n",
    "print(len(res_multi))\n",
    "assert len(glob.glob(output_path+'test2015/*.jpg')) == len(res_multi), f\"{len(glob.glob(output_path+'test2015/*.jpg'))} != {len(res_multi)}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a35f6ab-e880-4177-ba5a-0e2cfda9dbb4",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "img_id_to_annos_idx_map = {x[\"img_id\"]: i for i, x in enumerate(annos)}\n",
    "anno_list = json.load(open('../FGAHOI/data/hico_20160224_det/annotations/anno_list.json', 'r'))\n",
    "anno_list_to_idx_map = {x[\"global_id\"]: i for i, x in enumerate(anno_list)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "ea5b599a-ae5f-4fb5-9bb2-e3606ec462ae",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def listdir_nohidden(path, search=\"*\"):\n",
    "    return glob.glob(os.path.join(path, search))\n",
    "def convert_bbox(anno, new_width, new_height):\n",
    "    annotations = anno['annotations']\n",
    "    for an in annotations:\n",
    "        an['bbox'] = [new_width*an['bbox'][0]/anno['width'],\n",
    "                      new_height*an['bbox'][1]/anno['height'],\n",
    "                      new_width*an['bbox'][2]/anno['width'],\n",
    "                      new_height*an['bbox'][3]//anno['height']]\n",
    "        an['bbox'] = [int(a) for a in an['bbox']]\n",
    "    anno['width'] = new_width\n",
    "    anno['height'] = new_height\n",
    "def generate_new_anno(img_id, idx, annos):\n",
    "    assert idx in [0,1]\n",
    "    new_anno = deepcopy(annos[img_id_to_annos_idx_map[img_id]])\n",
    "    convert_bbox(new_anno, new_width=512, new_height=512)\n",
    "    return new_anno"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "2db68970-8634-40dd-95cc-61cf218fbbd1",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "new_annos_1 = []\n",
    "dst = new_dataset_path+'images/test2015'\n",
    "os.makedirs(os.path.dirname(dst), exist_ok=True)\n",
    "if not os.path.exists(dst):\n",
    "    os.symlink(os.path.abspath(output_path+'test2015/'), dst)\n",
    "new_anno_list_1 = deepcopy(anno_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "2c239fdc-86a8-487a-8d79-f348c2cd5392",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "assert len(glob.glob(output_path+'test2015/*.jpg')) == len(res_multi), f\"{len(glob.glob(output_path+'test2015/*.jpg'))} != {len(res_multi)}\"\n",
    "for anno in res_multi:\n",
    "    img_id = anno['img_id']\n",
    "    generated_imgs = listdir_nohidden(output_path+'test2015/', f'{anno[\"file_name\"]}')\n",
    "    if not len(generated_imgs) == 1:\n",
    "        print(f\"{img_id} {generated_imgs} {anno}\")\n",
    "        continue\n",
    "    new_annos_1.append(generate_new_anno(img_id, 0, annos))\n",
    "\n",
    "    t = new_anno_list_1[anno_list_to_idx_map[anno['file_name'].split('.')[0]]]\n",
    "\n",
    "    ori_width = t['image_size'][1]\n",
    "    ori_height = t['image_size'][0]\n",
    "    new_height, new_width = 512, 512\n",
    "    for hoi in t['hois']:\n",
    "        for i, bbox in enumerate(hoi['human_bboxes']):\n",
    "            bbox = [new_width*bbox[0]/ori_width,\n",
    "                    new_height*bbox[1]/ori_height,\n",
    "                    new_width*bbox[2]/ori_width,\n",
    "                    new_height*bbox[3]/ori_height]\n",
    "            bbox = [int(a) for a in bbox]\n",
    "            hoi['human_bboxes'][i] = bbox\n",
    "        for i, bbox in enumerate(hoi['object_bboxes']):\n",
    "            bbox = [new_width*bbox[0]/ori_width,\n",
    "                    new_height*bbox[1]/ori_height,\n",
    "                    new_width*bbox[2]/ori_width,\n",
    "                    new_height*bbox[3]/ori_height]\n",
    "            bbox = [int(a) for a in bbox]\n",
    "            hoi['object_bboxes'][i] = bbox\n",
    "    t['image_size'] = [new_height, new_width, 3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "23d1daab-153d-4d9e-89cb-543b9a7ae361",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "os.makedirs(os.path.dirname(new_dataset_path+'/annotations/'), exist_ok=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "5670be71-5f62-41bb-a117-ca549f4f399f",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "anno_list1_path = os.path.join(new_dataset_path, \"annotations/anno_list.json\")\n",
    "json.dump(new_anno_list_1, open(anno_list1_path, 'w'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "6416f578-96c2-4477-b37d-8696d8437501",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "anno1_path = os.path.join(new_dataset_path, \"annotations/test_hico.json\")\n",
    "os.makedirs(os.path.dirname(anno1_path), exist_ok=True)\n",
    "json.dump(new_annos_1, open(anno1_path, 'w'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "f33659f0-4c0f-4514-800a-e7ed48c1ac35",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# copy other annotations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "2d31bcb7-30fe-4970-90a6-b77670eb6783",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "to_copy_over = listdir_nohidden('../FGAHOI/data/hico_20160224_det/annotations/',\n",
    "                                search=\"[!tr|anno]*\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "69177f24-dc78-43d6-abd6-1e2697e0a3b1",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['FGAHOI/data/hico_20160224_det/annotations/hoi_list_new.json',\n",
       " 'FGAHOI/data/hico_20160224_det/annotations/corre_hico.npy',\n",
       " 'FGAHOI/data/hico_20160224_det/annotations/hoi_id_to_num.json',\n",
       " 'FGAHOI/data/hico_20160224_det/annotations/corre_hico_1.npy',\n",
       " 'FGAHOI/data/hico_20160224_det/annotations/file_name_to_obj_cat.json']"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "to_copy_over"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "78c45074-2bea-40ff-813b-55294961c99e",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "for file in to_copy_over:\n",
    "    shutil.copy2(file, new_dataset_path+'/annotations/')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "834b0b91-1dcd-476c-99c8-733eab30d1f9",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "file = '../FGAHOI/data/hico_20160224_det/annotations/trainval_hico.json'\n",
    "shutil.copy2(file, new_dataset_path+'/annotations/')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "fa675dd4-55f9-47bc-809c-55df8861628b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Done\n"
     ]
    }
   ],
   "source": [
    "print(\"Done\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8892b17-4132-46cc-8408-5582c2fd1be9",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
