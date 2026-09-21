#include <gst/gst.h>
#include <gst/webrtc/webrtc.h>

#include <iostream>
#include <cstring>
#ifdef _WIN32
#include <Windows.h>
#endif

static void print_module(const char* label, const char* module) {
#ifdef _WIN32
    const auto handle = GetModuleHandleA(module);
    char path[32768]{};
    const auto length = handle ? GetModuleFileNameA(handle, path, sizeof(path)) : 0;
    std::cerr << label << "=" << (length ? path : "<not loaded>") << "\n";
#else
    std::cerr << label << "=<platform unavailable>\n";
#endif
}

int main(int argc, char** argv) {
    gst_init(&argc, &argv);
    guint major{}, minor{}, micro{}, nano{};
    gst_version(&major, &minor, &micro, &nano);
    std::cerr << "GStreamer runtime=" << major << "." << minor << "." << micro << "." << nano << "\n";
    print_module("gstreamer_core_dll", "gstreamer-1.0-0.dll");
    print_module("glib_dll", "glib-2.0-0.dll");
    auto* webrtc_factory = gst_element_factory_find("webrtcbin");
    if (!webrtc_factory) { std::cerr << "webrtcbin factory lookup failed\n"; return 2; }
    auto* webrtc_plugin = gst_plugin_feature_get_plugin(GST_PLUGIN_FEATURE(webrtc_factory));
    std::cerr << "webrtcbin_plugin=" << (webrtc_plugin ? gst_plugin_get_filename(webrtc_plugin) : "<unknown>") << "\n";
    gst_object_unref(webrtc_factory);
    GstElement* pipeline = gst_pipeline_new("webrtc-runtime-probe");
    GstElement* source = gst_element_factory_make("videotestsrc", "source");
    GstElement* convert = gst_element_factory_make("videoconvert", "convert");
    GstElement* encoder = gst_element_factory_make("vp8enc", "encoder");
    GstElement* pay = gst_element_factory_make("rtpvp8pay", "pay");
    GstElement* capsfilter = gst_element_factory_make("capsfilter", "rtp_caps");
    GstElement* webrtc = gst_element_factory_make("webrtcbin", "webrtc");
    if (!pipeline || !source || !convert || !encoder || !pay || !capsfilter || !webrtc) {
        std::cerr << "missing required GStreamer factory\n";
        return 1;
    }
    GstCaps* rtp_caps = gst_caps_from_string("application/x-rtp,media=video,encoding-name=VP8,payload=96,clock-rate=90000");
    g_object_set(capsfilter, "caps", rtp_caps, nullptr);
    gst_caps_unref(rtp_caps);
    gst_bin_add_many(GST_BIN(pipeline), source, convert, encoder, pay, capsfilter, webrtc, nullptr);
    if (!gst_element_link_many(source, convert, encoder, pay, capsfilter, nullptr)) {
        std::cerr << "failed to link videotestsrc branch\n";
        gst_object_unref(pipeline);
        return 1;
    }
    // Request the transceiver-backed pad while the bin is still in NULL.
    // Passing caps to gst_element_request_pad can reject an otherwise valid
    // application/x-rtp request on some 1.26 builds; caps are enforced by the
    // upstream capsfilter instead.
    GstPadTemplate* sink_template = gst_element_class_get_pad_template(GST_ELEMENT_GET_CLASS(webrtc), "sink_%u");
    std::cerr << "webrtcbin_sink_template=" << (sink_template ? "present" : "missing") << "\n";
    GstPad* sink = gst_element_request_pad_simple(webrtc, "sink_%u");
    GstPad* src = gst_element_get_static_pad(capsfilter, "src");
    const auto link_result = sink && src ? gst_pad_link_full(src, sink, GST_PAD_LINK_CHECK_DEFAULT) : GST_PAD_LINK_REFUSED;
    if (!sink) std::cerr << "webrtcbin request_pad=sink_%u failed\n";
    if (!src) std::cerr << "capsfilter did not provide src pad\n";
    if (sink && src && link_result != GST_PAD_LINK_OK) std::cerr << "pad link result=" << gst_pad_link_get_name(link_result) << "\n";
    const bool linked = link_result == GST_PAD_LINK_OK;
    if (src) gst_object_unref(src);
    if (sink) gst_object_unref(sink);
    if (!linked) {
        std::cerr << "failed to request/link webrtcbin sink pad\n";
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
        return 1;
    }
    gst_element_set_state(pipeline, GST_STATE_READY);
    gst_element_get_state(pipeline, nullptr, nullptr, 5 * GST_SECOND);
    gst_element_set_state(pipeline, GST_STATE_PAUSED);
    GstBus* bus = gst_element_get_bus(pipeline);
    int result = 0;
    while (auto* message = gst_bus_timed_pop_filtered(bus, 500 * GST_MSECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS))) {
        if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
            GError* error = nullptr; gchar* debug = nullptr;
            gst_message_parse_error(message, &error, &debug);
            std::cerr << "GStreamer probe error: " << (error ? error->message : "unknown") << "\n"
                      << "debug: " << (debug ? debug : "") << "\n";
            g_clear_error(&error); g_free(debug); result = 1;
            gst_message_unref(message);
            break;
        }
        gst_message_unref(message);
    }
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    return result;
}
