#include "state/StandaloneSessionState.h"
#include <algorithm>
#include <cmath>

namespace rave {
namespace { bool finite(float v) { return std::isfinite(v); } }
StandaloneSessionState::StandaloneSessionState() { setLatentCount(0); }
void StandaloneSessionState::setLatentCount(std::size_t n) {
 n=std::min(n,maximumLatents); const juce::ScopedLock l(controlLock); auto old=count.load();
 auto nv=n?std::make_unique<std::atomic<float>[]>(n):nullptr; auto nm=std::make_unique<std::atomic<int>[]>(n+1);
 nm[0].store(mappings?mappings[0].load():-1); for(size_t i=0;i<n;++i){nv[i].store(i<old?latentValues[i].load():0);nm[i+1].store(i<old?mappings[i+1].load():-1);}
 latentValues=std::move(nv); mappings=std::move(nm); count.store(n,std::memory_order_release);
}
size_t StandaloneSessionState::latentCount()const noexcept{return count.load(std::memory_order_acquire);}
float StandaloneSessionState::latent(size_t i)const noexcept{return i<latentCount()?latentValues[i].load():0;}
bool StandaloneSessionState::setLatent(size_t i,float v)noexcept{if(i>=latentCount()||!finite(v))return false;latentValues[i].store(std::clamp(v,-4.f,4.f));return true;}
void StandaloneSessionState::setDryWet(float v)noexcept{if(finite(v))mix.store(std::clamp(v,0.f,1.f));}
float StandaloneSessionState::dryWet()const noexcept{return mix.load();}
void StandaloneSessionState::beginMidiLearn(int t)noexcept{if(t==dryWetTarget||(t>=0&&size_t(t)<latentCount()))learning.store(t);}
void StandaloneSessionState::clearMidiMapping(int t)noexcept{auto i=t==dryWetTarget?0:t+1;if(i>=0&&size_t(i)<=latentCount())mappings[i].store(-1);}
int StandaloneSessionState::midiController(int t)const noexcept{auto i=t==dryWetTarget?0:t+1;return i>=0&&size_t(i)<=latentCount()?mappings[i].load():-1;}
bool StandaloneSessionState::applyMidiCc(int cc,int value)noexcept{if(cc<0||cc>127||value<0||value>127)return false;auto learned=learning.exchange(-2);if(learned>=-1){for(size_t i=0;i<=latentCount();++i)if(mappings[i].load()==cc)mappings[i].store(-1);mappings[learned==dryWetTarget?0:size_t(learned+1)].store(cc);}bool hit=false;float n=float(value)/127.f;if(mappings[0].load()==cc){setDryWet(n);hit=true;}for(size_t i=0;i<latentCount();++i)if(mappings[i+1].load()==cc){setLatent(i,-4.f+8.f*n);hit=true;}return hit;}
void StandaloneSessionState::setModelPath(juce::String v){const juce::ScopedLock l(controlLock);identity.modelPath=v.substring(0,maximumStringLength);}
void StandaloneSessionState::setMidiInputId(juce::String v){const juce::ScopedLock l(controlLock);identity.midiInputId=v.substring(0,maximumStringLength);}
void StandaloneSessionState::setAudioSetup(juce::String t,juce::String o,juce::String i){const juce::ScopedLock l(controlLock);identity.audioDeviceType=t.substring(0,maximumStringLength);identity.audioOutputId=o.substring(0,maximumStringLength);identity.audioInputId=i.substring(0,maximumStringLength);}
StandaloneSessionState::Snapshot StandaloneSessionState::snapshot()const{const juce::ScopedLock l(controlLock);auto s=identity;s.dryWet=dryWet();s.latents.resize(latentCount());s.midiControllers.resize(latentCount()+1);s.midiControllers[0]=mappings[0].load();for(size_t i=0;i<latentCount();++i){s.latents[i]=latentValues[i].load();s.midiControllers[i+1]=mappings[i+1].load();}return s;}
bool StandaloneSessionState::restore(const Snapshot&s){if(s.latents.size()>maximumLatents||s.midiControllers.size()!=s.latents.size()+1||!finite(s.dryWet)||static_cast<std::size_t>(s.modelPath.length())>maximumStringLength||static_cast<std::size_t>(s.midiInputId.length())>maximumStringLength||static_cast<std::size_t>(s.audioDeviceType.length())>maximumStringLength||static_cast<std::size_t>(s.audioOutputId.length())>maximumStringLength||static_cast<std::size_t>(s.audioInputId.length())>maximumStringLength)return false;for(auto v:s.latents)if(!finite(v))return false;for(auto c:s.midiControllers)if(c < -1||c>127)return false;setLatentCount(s.latents.size());setDryWet(s.dryWet);for(size_t i=0;i<s.latents.size();++i)setLatent(i,s.latents[i]);{const juce::ScopedLock l(controlLock);identity=s;for(size_t i=0;i<s.midiControllers.size();++i)mappings[i].store(s.midiControllers[i]);}return true;}
bool StandaloneSessionState::serialize(juce::MemoryBlock&o)const{auto s=snapshot();juce::XmlElement x("RaveStandaloneState");x.setAttribute("version",schemaVersion);x.setAttribute("model",s.modelPath);x.setAttribute("midiInput",s.midiInputId);x.setAttribute("deviceType",s.audioDeviceType);x.setAttribute("output",s.audioOutputId);x.setAttribute("input",s.audioInputId);x.setAttribute("dryWet",s.dryWet);x.setAttribute("count",int(s.latents.size()));for(size_t i=0;i<s.latents.size();++i){x.setAttribute("v"+juce::String(i),s.latents[i]);x.setAttribute("cc"+juce::String(i+1),s.midiControllers[i+1]);}x.setAttribute("cc0",s.midiControllers[0]);auto text=x.toString();if(text.getNumBytesAsUTF8()>maximumSerializedBytes)return false;o.replaceAll(text.toRawUTF8(),text.getNumBytesAsUTF8());return true;}
bool StandaloneSessionState::deserialize(const void*d,size_t n){if(!d||n==0||n>maximumSerializedBytes)return false;auto x=juce::parseXML(juce::String::fromUTF8(static_cast<const char*>(d),int(n)));if(!x||!x->hasTagName("RaveStandaloneState")||x->getIntAttribute("version")!=schemaVersion)return false;Snapshot s;s.modelPath=x->getStringAttribute("model");s.midiInputId=x->getStringAttribute("midiInput");s.audioDeviceType=x->getStringAttribute("deviceType");s.audioOutputId=x->getStringAttribute("output");s.audioInputId=x->getStringAttribute("input");s.dryWet=float(x->getDoubleAttribute("dryWet"));auto c=x->getIntAttribute("count",-1);if(c<0||c>int(maximumLatents))return false;s.latents.resize(size_t(c));s.midiControllers.resize(size_t(c)+1);s.midiControllers[0]=x->getIntAttribute("cc0",-1);for(int i=0;i<c;++i){s.latents[size_t(i)]=float(x->getDoubleAttribute("v"+juce::String(i)));s.midiControllers[size_t(i)+1]=x->getIntAttribute("cc"+juce::String(i+1),-1);}return restore(s);}
}
