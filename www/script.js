class ImageUpload extends React.Component {
  constructor(props) {
    super(props);
    this.state = { file: '', imagePreviewUrl: '', message: '' };
  }

  _handleSubmit(e) {
    e.preventDefault();
    return new Promise((resolve, reject) => {
      let imageFormData = new FormData();
  
      imageFormData.append('image', this.state.file);
      
      var xhr = new XMLHttpRequest();
      var z = this;
      
      xhr.open('post', '/api/image', true);
      
      xhr.onload = function () {
        if (this.status == 200) {
          var msg = JSON.parse(this.response)
          z.setState({imagePreviewUrl:`data:image/jpeg;base64,${msg.pred.img}`})
          resolve(this.response);
        } else {
          reject(this.statusText);
        }
      };
      
      xhr.send(imageFormData);
  
    });
  }

  _handleImageChange(e) {
    e.preventDefault();

    let reader = new FileReader();
    let file = e.target.files[0];

    reader.onloadend = () => {
      this.setState({
        file: file,
        message: '',
        imagePreviewUrl: reader.result });
    };

    reader.readAsDataURL(file);
  }

  render() {
    let { imagePreviewUrl } = this.state;
    let $imagePreview = null;
    if (imagePreviewUrl) {
      $imagePreview = React.createElement("img", { className: "mx-auto d-block img-fluid", src: imagePreviewUrl });
    } else {
      $imagePreview = React.createElement("div", { className: "d-flex justify-content-center font-weight-normal" }, "Please select an Image for Preview");
    }

    return (
      React.createElement("div", { className: "previewComponent" },

      React.createElement("form", { onSubmit: e => this._handleSubmit(e) }),

      React.createElement("div", {className: "d-flex justify-content-center"},
        React.createElement("input", { className: "btn btn-primary btn-sm float-left",
          type: "file",
          id: "file",
          onChange: e => this._handleImageChange(e) })),

      React.createElement("div", {className: "d-flex justify-content-center"},
        React.createElement("button", { className: "btn btn-primary btn-lg submit-button",
          type: "submit",
          onClick: e => this._handleSubmit(e) }, "Generate Caption")),

      React.createElement("div", {}, $imagePreview),

      React.createElement("span", {}, this.state.message),
      )
      );
  }}


ReactDOM.render(React.createElement(ImageUpload, null), document.getElementById("mainApp"));