import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";


const FIELD_NAMES = [
    "ckpt_name",
    "system_msg",
    "prompt",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "frequency_penalty",
    "presence_penalty",
    "repeat_penalty",
    "seed",
    "sampling_mode",
    "thinking",
    "use_default_template",
];


function widgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}


function widgetValue(node, name) {
    const widget = widgetByName(node, name);
    return widget ? widget.value : undefined;
}


function linkedNodeForInput(node, inputName) {
    const input = node.inputs?.find((item) => item.name === inputName);
    if (!input?.link) {
        return null;
    }
    const link = app.graph?.links?.[input.link];
    if (!link) {
        return null;
    }
    return app.graph?.getNodeById?.(link.origin_id) || null;
}


function loaderConfigFromNode(node) {
    if (!node || node.comfyClass !== "LLMLoader") {
        return null;
    }
    return {
        ckpt_name: widgetValue(node, "ckpt_name"),
        max_ctx: widgetValue(node, "max_ctx"),
        gpu_layers: widgetValue(node, "gpu_layers"),
        n_threads: widgetValue(node, "n_threads"),
        use_mlock: widgetValue(node, "use_mlock"),
    };
}


function findLoaderForScratchpad(node) {
    const linked = linkedNodeForInput(node, "model");
    const linkedConfig = loaderConfigFromNode(linked);
    if (linkedConfig?.ckpt_name) {
        return { node: linked, config: linkedConfig };
    }

    const ckptName = widgetValue(node, "ckpt_name");
    const loaders = app.graph?._nodes?.filter((item) => item.comfyClass === "LLMLoader") || [];
    const matching = loaders.find((item) => widgetValue(item, "ckpt_name") === ckptName);
    const fallback = matching || (loaders.length === 1 ? loaders[0] : null);
    const fallbackConfig = loaderConfigFromNode(fallback);
    if (fallbackConfig?.ckpt_name) {
        return { node: fallback, config: fallbackConfig };
    }
    return null;
}


function setWidgetValue(node, name, value) {
    const widget = widgetByName(node, name);
    if (!widget) {
        return;
    }
    widget.value = value;
    if (widget.inputEl) {
        widget.inputEl.value = value;
    }
    app.canvas?.setDirty(true, true);
}


function disableSerialize(widget) {
    if (!widget.options) {
        widget.options = {};
    }
    widget.options.serialize = false;
}


function setButtonName(widget, name) {
    if (widget) {
        widget.name = name;
        app.canvas?.setDirty(true, true);
    }
}


function collectPayload(node) {
    const payload = {};
    for (const name of FIELD_NAMES) {
        payload[name] = widgetValue(node, name);
    }
    const loader = findLoaderForScratchpad(node);
    if (loader) {
        payload.loader_node_id = loader.node.id;
        payload.loader_config = loader.config;
    }
    return payload;
}


async function sendPrompt(node) {
    const sendButton = node.vlmnodesSendButton;
    const oldName = sendButton?.name || "Send";
    setButtonName(sendButton, "Sending...");

    try {
        const response = await api.fetchApi("/vlmnodes/llm_scratchpad/send", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(collectPayload(node)),
        });
        const result = await response.json();
        if (!response.ok || !result.ok) {
            if (result.model_status) {
                setWidgetValue(node, "model_status", JSON.stringify(result.model_status, null, 2));
            }
            throw new Error(result.error || response.statusText || "Request failed");
        }
        setWidgetValue(node, "response", result.text || "");
        if (result.model_status) {
            setWidgetValue(node, "model_status", JSON.stringify(result.model_status, null, 2));
        }
        setButtonName(sendButton, "Sent");
        setTimeout(() => setButtonName(sendButton, oldName), 1200);
    } catch (error) {
        setWidgetValue(node, "response", `Error: ${error.message || error}`);
        setButtonName(sendButton, "Error");
        setTimeout(() => setButtonName(sendButton, oldName), 1800);
    }
}


function copyWithFallback(text) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-9999px";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    let copied = false;
    try {
        copied = document.execCommand("copy");
    } finally {
        textArea.remove();
    }
    return copied;
}


async function copyResponse(node) {
    const copyButton = node.vlmnodesCopyButton;
    const oldName = copyButton?.name || "Copy";
    const text = widgetValue(node, "response") || "";
    if (!text) {
        setButtonName(copyButton, "Nothing to copy");
        setTimeout(() => setButtonName(copyButton, oldName), 1200);
        return;
    }

    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
        } else if (!copyWithFallback(text)) {
            throw new Error("Clipboard blocked");
        }
        setButtonName(copyButton, "Copied");
    } catch (error) {
        if (!copyWithFallback(text)) {
            setButtonName(copyButton, "Clipboard blocked");
            setTimeout(() => setButtonName(copyButton, oldName), 1800);
            return;
        }
        setButtonName(copyButton, "Copied");
    }
    setTimeout(() => setButtonName(copyButton, oldName), 1200);
}


app.registerExtension({
    name: "vlmnodes.LLMScratchpad",
    async nodeCreated(node) {
        if (node?.comfyClass !== "LLMScratchpad") {
            return;
        }
        if (!node.vlmnodesSendButton) {
            node.vlmnodesSendButton = node.addWidget("button", "Send", "", () => sendPrompt(node));
            disableSerialize(node.vlmnodesSendButton);
        }
        if (!node.vlmnodesCopyButton) {
            node.vlmnodesCopyButton = node.addWidget("button", "Copy", "", () => copyResponse(node));
            disableSerialize(node.vlmnodesCopyButton);
        }
    },
});
