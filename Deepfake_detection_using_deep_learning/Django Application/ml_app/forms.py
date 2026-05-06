from django import forms

class VideoUploadForm(forms.Form):
    upload_video_file = forms.FileField(
        label="Select Video",
        required=True,
        widget=forms.FileInput(attrs={
            "accept": "video/*",
            "id": "video-file-input",
            "class": "file-input",
        })
    )
    sequence_length = forms.IntegerField(
        label="Sequence Length",
        required=False,
        initial=20,
        widget=forms.HiddenInput(attrs={"id": "id_sequence_length"})
    )

class ImageUploadForm(forms.Form):
    upload_image_file = forms.FileField(
        label="Select Image",
        required=True,
        widget=forms.FileInput(attrs={
            "accept": "image/*",
            "id": "image-file-input",
            "class": "file-input",
        })
    )
