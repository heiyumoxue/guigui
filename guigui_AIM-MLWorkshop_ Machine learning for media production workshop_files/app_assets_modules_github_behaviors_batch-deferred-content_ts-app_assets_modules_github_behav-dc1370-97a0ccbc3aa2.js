"use strict";(globalThis.webpackChunk=globalThis.webpackChunk||[]).push([["app_assets_modules_github_behaviors_batch-deferred-content_ts-app_assets_modules_github_behav-dc1370"],{47442(a,b,c){c.d(b,{s:()=>BaseBatchDeferredContentElement});var d=c(76006),e=c(75662);class BatchLoader{loadInBatch(a){let b=this.autoFlushingQueue.push(a);return new Promise(a=>this.callbacks.set(b,a))}async load(a){let b=new Map;for(let[c,d]of a)b.set(d,c);let e=new FormData;for(let[f,g]of(e.set("_method","GET"),b.entries()))for(let h of g.inputs)e.append(`items[${f}][${h.name}]`,h.value);let i=await fetch(this.url,{method:"POST",body:e,headers:{Accept:"application/json","X-Requested-With":"XMLHttpRequest"}});if(i.ok){let j=await i.json();if(!j||"object"!=typeof j||Array.isArray(j))throw Error("Malformed batch response");for(let k in j){let l=this.callbacks.get(k);if(l){let m=j[k];this.validate(m),l(m)}}}}constructor(a,b){this.url=a,this.callbacks=new Map,this.autoFlushingQueue=new class AutoFlushingQueue{push(a){let b=`item-${this.index++}`;return this.timer&&(window.clearTimeout(this.timer),this.timer=null),this.elements.length>=this.limit&&this.flush(),this.timer=window.setTimeout(()=>{this.timer=null,this.flush()},this.timeout),this.elements.push([a,b]),b}onFlush(a){this.callbacks.push(a)}async flush(){let a=this.elements.splice(0,this.limit);0!==a.length&&await Promise.all(this.callbacks.map(b=>b(a)))}constructor(a=50,b=30){this.elements=[],this.timer=null,this.callbacks=[],this.timeout=a,this.limit=b,this.index=0}},this.autoFlushingQueue.onFlush(async a=>{this.load(a)}),this.validate=b}}var f=function(a,b,c,d){var e,f=arguments.length,g=f<3?b:null===d?d=Object.getOwnPropertyDescriptor(b,c):d;if("object"==typeof Reflect&&"function"==typeof Reflect.decorate)g=Reflect.decorate(a,b,c,d);else for(var h=a.length-1;h>=0;h--)(e=a[h])&&(g=(f<3?e(g):f>3?e(b,c,g):e(b,c))||g);return f>3&&g&&Object.defineProperty(b,c,g),g};class BaseBatchDeferredContentElement extends HTMLElement{async connectedCallback(){let a=await this.batchLoader.loadInBatch(this);this.update(a)}get batchLoader(){let a=this.getAttribute("data-url");if(!a)throw Error(`${this.tagName} element requires a data-url attribute`);let b=this.batchLoaders.get(a);return b||(b=new BatchLoader(a,a=>this.validate(a)),this.batchLoaders.set(a,b)),b}}let g=new Map,h=class BatchDeferredContentElement extends BaseBatchDeferredContentElement{validate(a){if("string"!=typeof a)throw Error("Batch deferred content was not a string")}update(a){let b=(0,e.r)(document,a);this.replaceWith(b)}constructor(...a){super(...a),this.batchLoaders=g}};f([d.GO],h.prototype,"inputs",void 0),h=f([d.Ih],h)},86975(a,b,c){c.d(b,{AU:()=>o,DT:()=>s,F2:()=>m,HN:()=>k,Lq:()=>i,Si:()=>n,T2:()=>y,Yg:()=>x,aN:()=>l,ag:()=>w,po:()=>v,q3:()=>p,uL:()=>z,wz:()=>r,xc:()=>j,xk:()=>t,zH:()=>h});var d=c(74395),e=c(64707);let f=d.session.adapter,g="data-turbo-loaded";function h(){document.documentElement.setAttribute(g,"")}function i(){return document.documentElement.hasAttribute(g)}let j=()=>!(0,e.c)("PJAX_ENABLED"),k=a=>a?.tagName==="TURBO-FRAME",l=()=>{f.progressBar.setValue(0),f.progressBar.show()},m=()=>{f.progressBar.setValue(1),f.progressBar.hide()},n=(a,b)=>{let c=new URL(a,window.location.origin),d=new URL(b,window.location.origin);return Boolean(d.hash)&&c.host===d.host&&c.pathname===d.pathname&&c.search===d.search};function o(a,b){let c=a.split("/",3).join("/"),d=b.split("/",3).join("/");return c===d}async function p(){let a=document.head.querySelectorAll("link[rel=stylesheet]"),b=new Set([...document.styleSheets].map(a=>a.href)),c=[];for(let d of a)""===d.href||b.has(d.href)||c.push(q(d));await Promise.all(c)}let q=(a,b=2e3)=>new Promise(c=>{let d=()=>{a.removeEventListener("error",d),a.removeEventListener("load",d),c()};a.addEventListener("load",d,{once:!0}),a.addEventListener("error",d,{once:!0}),setTimeout(d,b)}),r=(a,b)=>{let c=b||a.querySelectorAll("[data-turbo-replace]"),d=[...document.querySelectorAll("[data-turbo-replace]")];for(let e of c){let f=d.find(a=>a.id===e.id);f&&f.replaceWith(e)}},s=a=>{for(let b of a.querySelectorAll("link[rel=stylesheet]"))document.head.querySelector(`link[href="${b.getAttribute("href")}"],
           link[data-href="${b.getAttribute("data-href")}"]`)||document.head.append(b)},t=a=>{for(let b of a.querySelectorAll("script"))document.head.querySelector(`script[src="${b.getAttribute("src")}"]`)||u(b)},u=a=>{let{src:b}=a,c=document.createElement("script"),d=a.getAttribute("type");d&&(c.type=d),c.src=b,document.head&&document.head.appendChild(c)},v=a=>{let b=[];for(let c of a.querySelectorAll('meta[data-turbo-track="reload"]'))document.querySelector(`meta[http-equiv="${c.getAttribute("http-equiv")}"]`)?.content!==c.content&&b.push(y(c.getAttribute("http-equiv")));return b},w=a=>{let b=a.querySelector("[data-turbo-head]")||a.head;return{title:b.querySelector("title")?.textContent,transients:[...b.querySelectorAll("[data-turbo-transient]")],bodyClasses:a.querySelector("meta[name=turbo-body-classes]")?.content,replacedElements:[...a.querySelectorAll("[data-turbo-replace]")]}},x=()=>[...document.documentElement.attributes],y=a=>a.replace(/^x-/,"").replaceAll("-","_"),z=a=>document.dispatchEvent(new CustomEvent("turbo:reload",{detail:{reason:a}}))},64707(a,b,c){c.d(b,{"$":()=>g,c:()=>f});var d=c(15205);let e=(0,d.Z)(function(){return(document.head?.querySelector('meta[name="enabled-features"]')?.content||"").split(",")}),f=(0,d.Z)(function(a){return -1!==e().indexOf(a)}),g={isFeatureEnabled:f}},86702(a,b,c){c.d(b,{Z:()=>e,"_":()=>f});var d=c(94109);class NoOpStorage{getItem(){return null}setItem(){}removeItem(){}clear(){}key(){return null}get length(){return 0}}function e(a,b={throwQuotaErrorsOnSet:!1},c=d.iG,e=a=>a,f=a=>a){let g;try{if(!c)throw Error();g=c[a]}catch{g=new NoOpStorage}let{throwQuotaErrorsOnSet:h}=b;return{getItem:function(a){try{let b=g.getItem(a);return b?e(b):null}catch(c){return null}},setItem:function(a,b){try{g.setItem(a,f(b))}catch(c){if(h&&c.message.toLowerCase().includes("quota"))throw c}},removeItem:function(a){try{g.removeItem(a)}catch(b){}}}}function f(a){return e(a,{throwQuotaErrorsOnSet:!1},window,JSON.parse,JSON.stringify)}},25811(a,b,c){c.d(b,{LS:()=>f,cl:()=>g,rV:()=>e});var d=c(86702);let{getItem:e,setItem:f,removeItem:g}=(0,d.Z)("sessionStorage")},55065(a,b,c){c.d(b,{"$g":()=>SoftNavSuccessEvent,OV:()=>SoftNavStartEvent,QW:()=>SoftNavErrorEvent,Xr:()=>SoftNavEndEvent});var d=c(45586);class SoftNavEvent extends Event{constructor(a,b){super(b),this.mechanism=a}}class SoftNavStartEvent extends SoftNavEvent{constructor(a){super(a,d.QE.START)}}class SoftNavSuccessEvent extends SoftNavEvent{constructor(a,b){super(a,d.QE.SUCCESS),this.visitCount=b}}class SoftNavErrorEvent extends SoftNavEvent{constructor(a,b){super(a,d.QE.ERROR),this.error=b}}class SoftNavEndEvent extends SoftNavEvent{constructor(a){super(a,d.QE.END)}}},45586(a,b,c){c.d(b,{jN:()=>j,QE:()=>i,FP:()=>p,r_:()=>o,Yl:()=>l,TL:()=>q,LD:()=>m,BT:()=>n,u5:()=>r});var d=c(55065),e=c(34855),f=c(58843);let g="stats:soft-nav-duration",h={turbo:"TURBO",react:"REACT","turbo.frame":"FRAME",ui:"UI"},i=Object.freeze({INITIAL:"soft-nav:initial",START:"soft-nav:start",SUCCESS:"soft-nav:success",ERROR:"soft-nav:error",FRAME_UPDATE:"soft-nav:frame-update",END:"soft-nav:end",RENDER:"soft-nav:render",PROGRESS_BAR:{START:"soft-nav:progress-bar:start",END:"soft-nav:progress-bar:end"}}),j="reload",k=0;function l(){k=0,document.dispatchEvent(new Event(i.INITIAL)),(0,f.XL)()}function m(a){(0,f.sj)()||(s(),document.dispatchEvent(new d.OV(a)),(0,f.U6)(a),(0,f.J$)(),(0,f.Nt)(),performance.mark(g))}function n(a={}){u(a)&&(k+=1,document.dispatchEvent(new d.$g((0,f.Gj)(),k)),p(a))}function o(a={}){var b;if(!u(a))return;k=0;let c=(0,f.Wl)()||j;document.dispatchEvent(new d.QW((0,f.Gj)(),c)),t(),b=c,(0,e.b)({turboFailureReason:b,turboStartUrl:(0,f.wP)(),turboEndUrl:window.location.href}),(0,f.XL)()}function p(a={}){u(a)&&(t(),document.dispatchEvent(new d.Xr((0,f.Gj)())),(0,f.pS)())}function q(a={}){u(a)&&(!function(){let a=function(){if(0===performance.getEntriesByName(g).length)return null;performance.measure(g,g);let a=performance.getEntriesByName(g),b=a.pop();return b?b.duration:null}();if(!a)return;let b=h[(0,f.Gj)()],c=Math.round(a);b===h.react&&document.dispatchEvent(new CustomEvent("staffbar-update",{detail:{duration:c}})),(0,e.b)({requestUrl:window.location.href,softNavigationTiming:{mechanism:b,destination:(0,f.Nb)()||"rails",duration:c,initiator:(0,f.CI)()||"rails"}})}(),document.dispatchEvent(new Event(i.RENDER)))}function r(){document.dispatchEvent(new Event(i.FRAME_UPDATE))}function s(){document.dispatchEvent(new Event(i.PROGRESS_BAR.START))}function t(){document.dispatchEvent(new Event(i.PROGRESS_BAR.END))}function u({skipIfGoingToReactApp:a,allowedMechanisms:b=[]}={}){return(0,f.sj)()&&(0===b.length||b.includes((0,f.Gj)()))&&(!a||!(0,f.Nb)())}},58843(a,b,c){c.d(b,{Ak:()=>r,CI:()=>v,Gj:()=>o,"J$":()=>u,Nb:()=>w,Nt:()=>s,OE:()=>p,U6:()=>l,Wl:()=>q,XL:()=>k,pS:()=>m,sj:()=>n,wP:()=>t});var d=c(25811),e=c(45586);let f="soft-nav:fail",g="soft-nav:fail-referrer",h="soft-nav:referrer",i="soft-nav:marker",j="soft-nav:react-app-name";function k(){(0,d.LS)(i,"0"),(0,d.cl)(h),(0,d.cl)(f),(0,d.cl)(g),(0,d.cl)(j)}function l(a){(0,d.LS)(i,a)}function m(){(0,d.LS)(i,"0")}function n(){let a=(0,d.rV)(i);return a&&"0"!==a}function o(){return(0,d.rV)(i)}function p(){return Boolean(q())}function q(){return(0,d.rV)(f)}function r(a){(0,d.LS)(f,a||e.jN),(0,d.LS)(g,window.location.href)}function s(){(0,d.LS)(h,window.location.href)}function t(){return(0,d.rV)(h)||document.referrer}function u(){let a=w();a?(0,d.LS)(j,a):(0,d.cl)(j)}function v(){return(0,d.rV)(j)}function w(){return document.querySelector('meta[name="ui"]')?"ui":document.querySelector("react-app")?.getAttribute("app-name")}},34855(a,b,c){c.d(b,{b:()=>g});var d=c(94109),e=c(80721);let f=[];function g(a,b=!1){void 0===a.timestamp&&(a.timestamp=new Date().getTime()),a.loggedIn=k(),a.staff=l(),f.push(a),b?j():i()}let h=null;async function i(){await e.C,null==h&&(h=window.requestIdleCallback(j))}function j(){if(h=null,!f.length)return;let a=d.n4?.head?.querySelector('meta[name="browser-stats-url"]')?.content;if(!a)return;let b=JSON.stringify({stats:f});try{navigator.sendBeacon&&navigator.sendBeacon(a,b)}catch{}f=[]}function k(){return!!d.n4?.head?.querySelector('meta[name="user-login"]')?.content}function l(){return!!d.n4?.head?.querySelector('meta[name="user-staff"]')?.content}d.n4?.addEventListener("pagehide",j),d.n4?.addEventListener("visibilitychange",j)}}])
//# sourceMappingURL=app_assets_modules_github_behaviors_batch-deferred-content_ts-app_assets_modules_github_behav-dc1370-890c59cdbb6b.js.map