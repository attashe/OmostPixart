
function buildPrompt() {
  if (globalState.seedMode === "random") {
    globalState.seedValue = getSeed();
    document.getElementById("seed").value = globalState.seedValue;
  }
  let seed = globalState.seedValue;
  prompt = {};

  let box_idx = 0;
  
  prompt['bboxes'] = [];
  let tags = new Array();
  AppSurface.shapes.forEach((box, boxes, map) => {
    console.log(box);
    if (box.caption) tags.push(box.caption);
    prompt['bboxes'][String(box_idx)] = {
      'caption': box.caption,
      'width': box.width,
      'height': box.height,
      'x': box.x,
      'y': box.y,
      'idx': box_idx,
    };
    box_idx += 1;
  });

  tags = ''//tags.join(";");
  let positive_prompt = `${globalState.positivePrompt.replace(
    /[\s;]+$/g,
    ""
  )};${tags}`;

  prompt['positive'] = positive_prompt
  prompt['negative'] = globalState.negativePrompt

  let canvas = document.getElementById("main-canvas");
  
  prompt['width'] = canvas.width
  prompt['height'] = canvas.height

  prompt['seed'] = seed
  prompt['steps'] = globalState.stepsValue
  prompt['cfg'] = globalState.cfgValue
  prompt['sampler_name'] = globalState.samplerName
  prompt['scheduler_name'] = globalState.schedulerName
  prompt['lora'] = globalState.loraMap

  console.log(prompt);
  // export2txt(prompt)
  return prompt;
}

var socketInitialized = false;
const clientID = "1122";
function onSocketMessageReceive(event) {
  try {
    let parsed = JSON.parse(event.data);
    if (parsed.type === "progress") {
      let progress = Math.round((100 * parsed.data.value) / parsed.data.max);
      document.getElementById("progress-bar").style.width = `${progress}%`;
    } else if (parsed.type === "status" && !parsed.data.sid) {
      if (parsed.data.status.exec_info.queue_remaining === 0) {
        requestGET("/history", (endpoint, response) => {
          if (globalState.promptID) {
            let pid = response[globalState.promptID];
            let images = pid.outputs[globalState.outputImageNode].images;
            images.forEach((image) => {
              let img_url = `/view?filename=${image.filename}&subfolder=${image.subfolder}&type=${image.type}`;
              getImage(img_url);
            });
          }
        });
      }
    }
  } catch (error) {}
}

function onSocketImageReceive(event) {
  let parsed = JSON.parse(event.data);
  let img_url = `/get_image/${parsed.data.image_id}`;
  getImage(img_url);
}

function initializeWebSocket() {
  if (socketInitialized) return;

  socketInitialized = true;

  // const socket = new WebSocket();
  const socket = io();

  socket.on("open", (event) => {
    console.log("Socket opened");
  });

  socket.on("message", onSocketMessageReceive);
  socket.on("update_image", onSocketImageReceive);
}

function queuePrompt() {
  let pb = document.getElementById("progress-bar");
  pb.style.width = "0%";
  let prompt = buildPrompt();
  initializeWebSocket();
  requestPOST(
    "/generate_image",
    {
      prompt: prompt,
      client_id: clientID,
    },
    (endpoint, response) => {
      console.log('Response:', response)
      if (response.error) {
        addToast(
          "<u>Oops</u>",
          response.error.message,
          (is_error = true),
          (timeout = 0)
        );
        let node_errors = response.node_errors;
        if (node_errors) {
          let node;
          for (var node_id in node_errors) {
            node = node_errors[node_id];
            console.log(node);
            let class_type = node.class_type;
            let errors = node.errors;
            for (var eid in errors) {
              console.log(errors[eid].message);
              console.log(errors[eid].details);
              addToast(
                `<u>Error in ${class_type}</u>`,
                `${errors[eid].message}, ${errors[eid].details}`,
                (is_error = true),
                (timeout = 0)
              );
            }
          }
        }
      }
      if (response.image_id) {
        addToast("Success!", "The prompt was queued succesfully.");
        globalState.promptID = response.prompt_id;
        console.log("prompt_id = ", globalState.promptID);
        getImage(`/get_image/${response.image_id}`);
      }
    }
  );
}
