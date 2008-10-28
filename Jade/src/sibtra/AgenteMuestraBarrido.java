/**
 * 
 */
package sibtra;

import jade.core.AID;
import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.domain.DFService;
import jade.domain.FIPAException;
import jade.domain.FIPANames;
import jade.domain.FIPAAgentManagement.DFAgentDescription;
import jade.domain.FIPAAgentManagement.ServiceDescription;
import jade.lang.acl.ACLMessage;
import jade.lang.acl.UnreadableException;
import jade.proto.SimpleAchieveREInitiator;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.HeadlessException;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import sibtra.lms.BarridoAngular;
import sibtra.lms.PanelMuestraBarrido;
import sibtra.lms.ZonaLMS;

/**
 * Este agente pedirá los barridos al AgenteLMS y los presenta en PanelMuestraBarrido
 * 
 * @author alberto
 *
 */
public class AgenteMuestraBarrido extends Agent {
	
	/**
	 * Ventana que contendrá el PanelMuestraBarrido
	 */
	protected agenteMBGUI miGui=null;
	
	/** Lista de agentes encontrados */
	protected AID[] AgentesLMS=null;

	protected ComportaEscaneoContinuo compEscaneoContinuo;
	
	protected ComportaPideBarrido compPideBarrido;
	
	protected ComportaPideConfiguracion compPideConfiguracion;

	private boolean logs;
	
	/**
	 * Metodo para emitir los mensajes de Log
	 */
	protected void agentLog(String m) {
		if(logs)
			System.out.println(getAID().getName()+
					": "+System.currentTimeMillis()+": "+m);
	}

	/**
	 * Inicialización:
	 * <ol>
	 *   <li>Creamos ventana gráfica y la mostramos
	 *   <li>Se registra en el DF
	 *   <li>Espara un rato
	 *   <li>Arranca en comportamiento {@link ComportaEscaneoContinuo}
	 * </ol> 
	 *  
	 * @see jade.core.Agent#setup()
	 */
	@Override
	protected void setup() {
		//super.setup();
		agentLog("Arranca");
		
		logs=true;
		//analizamos argumentos
		Object[] args = getArguments();
		if (args!=null) 
			for(int i=0; i<args.length; i++) {
				String aa=args[i].toString();
				agentLog("Argumento "+ i + " es '"+aa+"'");
				if(aa.startsWith("logs:") && aa.length()>"logs:".length() ) {
					logs=aa.substring(aa.indexOf(":")+1).equalsIgnoreCase("on");
				}
			}

		
		//Nos registramos en el DF
		DFAgentDescription  dfd = new DFAgentDescription();
		dfd.setName(getAID());
		ServiceDescription  sd = new ServiceDescription();
		sd.setType("muestra-barrido");
		sd.setName("mostrador-barridos");
		dfd.addServices(sd);
		try {
			DFService.register(this, dfd);
		}
		catch (FIPAException fe) {
			fe.printStackTrace();
		}

		//Creamos todos los comportamientos
		compPideBarrido=new ComportaPideBarrido();
		compPideConfiguracion=new ComportaPideConfiguracion();
		compEscaneoContinuo=new ComportaEscaneoContinuo();
		
		//Creamos la ventana grafica y la mostramos
		miGui=new agenteMBGUI(getLocalName());

		//Por defecto estamos en modo no continuo.
		//el GUI añadirá los comportamientos.
	}

	/**
	 * Teminación del agente.
	 * Se desregistra y cierra ventana si existe.
	 * @see jade.core.Agent#takeDown()
	 */
	@Override
	protected void takeDown() {
		// Deregister from the yellow pages
		try {
			DFService.deregister(this);
		}
		catch (FIPAException fe) {
			fe.printStackTrace();
		}

		// Close the GUI
		if(miGui != null ){
			agentLog("Cerramos GUI");
			miGui.dispose();
		}
		// Printout a dismissal message
		agentLog("terminado.");

	}
	
	/**
	 * Devuleve el primer agente LMS de la lista. 
	 * Si no hay ninguno trata de consegirlo.
	 * @return null si no hay, caso contrario el primer agente LMS
	 */
	protected AID agenteLMS() {
		// Obetenemos la lista de agentes generadores si no hay aún
		if (AgentesLMS==null || AgentesLMS.length==0) {
			DFAgentDescription template = new DFAgentDescription();
			ServiceDescription sd = new ServiceDescription();
			sd.setType("comunica-LMS");
			template.addServices(sd);
			try {
				DFAgentDescription[] result = DFService.search(AgenteMuestraBarrido.this, template);
				String s=new String("Encontramos los siguientes agentes LMS:");
				AgentesLMS = new AID[result.length];	
				for (int i = 0; i < result.length; ++i) {
					AgentesLMS[i] = result[i].getName();
					s+="\n\t"+AgentesLMS[i].getName();
				}
				agentLog(s);
			}
			catch (FIPAException fe) {
				fe.printStackTrace();
			}
		}
		if(AgentesLMS!=null && AgentesLMS.length>=1)
			return AgentesLMS[0];
		else
			return null;
	}
	
	//Calse privada con el GUI
	/**
	 * Muestra la ventana con el panel para mostrar los barrido y demás controles
	 */
	private class agenteMBGUI extends JFrame implements ActionListener, ChangeListener {

		/**
		 * Panel que muestra los barridos.
		 */
		PanelMuestraBarrido panBarr;

		SpinnerNumberModel modelPromedios;
		SpinnerNumberModel modelAngIni;
		SpinnerNumberModel modelAngFin;

		/**
		 * Etiqueta donde se inica la configuración obtenida
		 */
		JLabel labelConfiguración;

		/**
		 * Modelo del spiner de frecuencia de barrido continuo
		 */
		private SpinnerNumberModel modelFrecEscaneo;

		/**
		 * Bonton para pedir barridos manualmente
		 */
		private JButton jbPideBarrido;

		/**
		 * Boton donde se marca si se está haciendo escaneo continuo
		 */
		private JCheckBox cbEscaneo;

		/**
		 * Etiqueta de "Esperando" que se activará cuando se espera un barrido
		 */
		private JLabel jlEsperando;

		
		private static final String comandoBotonConfiguracion="Configuracion";
		private static final String comandoBotonPideBarrido="PideBarrido";
		private static final String comandoBotonEscaneoContinuo="EscaneoContinuo";
		private static final String comandoBotonZonas="ActulizaZonas";
		
		/**
		 * Añadimos los elementos que quermos que aparezcan.
		 * @param title Titulo
		 * @throws HeadlessException
		 */
		public agenteMBGUI(String title) throws HeadlessException {
			super(title);

			addWindowListener(new   WindowAdapter() {
				public void windowClosing(WindowEvent e) {
					AgenteMuestraBarrido.this.doDelete();
				}
			} );
			panBarr=new PanelMuestraBarrido((short)80);
			add(panBarr,BorderLayout.CENTER);

			//ponemos las dos líneas (spiners y configuración) en un box layout
			JPanel jpSur=new JPanel();
			jpSur.setLayout(new BoxLayout(jpSur,BoxLayout.PAGE_AXIS));

			//Los controles de promedios y angulos como spiner
			//dentro de un panel con un flow layout
			{
				JPanel jpSpinersBarrido;
				jpSpinersBarrido=new JPanel(new FlowLayout(FlowLayout.CENTER,10,4));

				JPanel jpa;

				modelPromedios=new SpinnerNumberModel(1,1,250,1);
				jpa=new JPanel();
				jpa.add(new JLabel("Promedios"));
				jpa.add(new JSpinner(modelPromedios));
				jpa.setBorder(BorderFactory.createLineBorder(Color.BLACK));
				jpSpinersBarrido.add(jpa);

				modelAngIni=new SpinnerNumberModel(0,0,180,1);
				modelAngIni.addChangeListener(this);
				jpa=new JPanel();
				jpa.add(new JLabel("Angulo Inicial"));
				jpa.add(new JSpinner(modelAngIni));
				jpa.setBorder(BorderFactory.createLineBorder(Color.BLACK));
				jpSpinersBarrido.add(jpa);

				modelAngFin=new SpinnerNumberModel(180,0,180,1);
				modelAngFin.addChangeListener(this);
				jpa=new JPanel();
				jpa.add(new JLabel("Angulo Final"));
				jpa.add(new JSpinner(modelAngFin));
				jpa.setBorder(BorderFactory.createLineBorder(Color.BLACK));
				jpSpinersBarrido.add(jpa);
				
				jpSur.add(jpSpinersBarrido);

			}

			jpSur.add(Box.createRigidArea(new Dimension(0,5))); //separador de líneas
			
			{  //Siguiente linea con lo relativo a cundo se realiza escaneo

				JPanel jpCuandoBarrido=new JPanel(new FlowLayout(FlowLayout.CENTER,10,4));
				//Check Box selección escaneo continulo
				cbEscaneo=new JCheckBox("Escaneo Continuo");
				cbEscaneo.setActionCommand(comandoBotonEscaneoContinuo);
				cbEscaneo.addActionListener(this);
				jpCuandoBarrido.add(cbEscaneo);

				//Spiner frecuencia escaneo continuo
				modelFrecEscaneo=new SpinnerNumberModel(1000,50,60000,5);
				modelFrecEscaneo.addChangeListener(this);
				JPanel jpa=new JPanel();
				jpa.add(new JLabel("Mili esc"));
				jpa.add(new JSpinner(modelFrecEscaneo));
				jpa.setBorder(BorderFactory.createLineBorder(Color.BLACK));
				jpCuandoBarrido.add(jpa);
				
				//Boton para pedir barrido
				jbPideBarrido=new JButton("Pide Barrido");
				jbPideBarrido.setActionCommand(comandoBotonPideBarrido);
				jbPideBarrido.addActionListener(this);
				jpCuandoBarrido.add(jbPideBarrido);
				
				jlEsperando=new JLabel("Esperando datos");
				jlEsperando.setEnabled(false);
				jpCuandoBarrido.add(jlEsperando);

				jpSur.add(jpCuandoBarrido);
			}
			
			jpSur.add(Box.createRigidArea(new Dimension(0,5))); //separador de líneas
			
			//La etiqueta de la configuración con el botón correspondiente
			{
				JPanel jpConfiguracion=new JPanel(new FlowLayout(FlowLayout.CENTER,10,4));
				labelConfiguración=new JLabel("(configuracion  :rango-angular ??? :en-milimetros ? :resolucion-angular ?? :codigo-rango ? )");
				jpConfiguracion.add(labelConfiguración);
				JButton botPideConfig=new JButton("Actualiza");
				botPideConfig.setActionCommand(comandoBotonConfiguracion);
				botPideConfig.addActionListener(this);
				jpConfiguracion.add(botPideConfig);
				jpSur.add(jpConfiguracion);
			}
			
			jpSur.add(Box.createRigidArea(new Dimension(0,5))); //separador de líneas
			
			
			{ //Elecciónn de conjunto de zonas y boton para la actualizacion
				JPanel jpZonas=new JPanel(new FlowLayout(FlowLayout.CENTER,10,4));
				jpZonas.add(new JLabel("Zonas"));
				JButton botPideZonas=new JButton("Actualiza");
				botPideZonas.setActionCommand(comandoBotonZonas);
				botPideZonas.addActionListener(this);
				jpZonas.add(botPideZonas);
				jpSur.add(jpZonas);
			}
			
			add(jpSur,BorderLayout.SOUTH);
			
			setSize(800, 400);
			setVisible(true);
		}

		/* (non-Javadoc)
		 * @see java.awt.event.ActionListener#actionPerformed(java.awt.event.ActionEvent)
		 */
		public void actionPerformed(ActionEvent e) {
			if(comandoBotonConfiguracion.equals(e.getActionCommand())) {
				compPideConfiguracion.reset();
				AgenteMuestraBarrido.this.addBehaviour(compPideConfiguracion);
			} else if(comandoBotonZonas.equals(e.getActionCommand())) {
				AgenteMuestraBarrido.this.addBehaviour(new ComportaPideZonas(true,ZonaLMS.ZONA_A));
				AgenteMuestraBarrido.this.addBehaviour(new ComportaPideZonas(true,ZonaLMS.ZONA_B));
				AgenteMuestraBarrido.this.addBehaviour(new ComportaPideZonas(true,ZonaLMS.ZONA_C));
			} else if(comandoBotonPideBarrido.equals(e.getActionCommand())) {
				AgenteMuestraBarrido.this.removeBehaviour(compEscaneoContinuo);
				compPideBarrido.reset();
				AgenteMuestraBarrido.this.addBehaviour(compPideBarrido);
			} else if(comandoBotonEscaneoContinuo.equals(e.getActionCommand())) {
				if(((JCheckBox)e.getSource()).isSelected()) { 
					jbPideBarrido.setEnabled(false);
					compEscaneoContinuo.reset();
					AgenteMuestraBarrido.this.addBehaviour(compEscaneoContinuo);
				} else {
					jbPideBarrido.setEnabled(true);
					AgenteMuestraBarrido.this.removeBehaviour(compEscaneoContinuo);					
				}
			}
		}

		/**
		 * establece el barrido pasado en {@link PanelMuestraBarrido} y programa la actualización
		 * @param om Barrido angular recibido
		 */
		public void actualizaBarrido(BarridoAngular om) {
			panBarr.setBarrido(om);
			//programamos la actualizacion de la ventana
			SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					panBarr.repaint();						
				}
			});

		}

		/**
		 * establece el la zona obtenida y programa la actualización
		 * @param om objeto de tipo ZonaLMS
		 */
		public void actualizaZona(ZonaLMS om) {
			panBarr.setZona(om);
			SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					panBarr.repaint();						
				}
			});			
		}

		/**
		 * Para controlar que el angulo inicial sea siempre <= que el final
		 * @see javax.swing.event.ChangeListener#stateChanged(javax.swing.event.ChangeEvent)
		 */
		public void stateChanged(ChangeEvent even) {
			//actuamos si hay problema
			if(modelAngIni.getNumber().intValue() > modelAngFin.getNumber().intValue()) {
				//el que se está moviendo, manad
				if(even.getSource().equals(modelAngIni))
					modelAngFin.setValue(modelAngIni.getNumber());
				else
					modelAngIni.setValue(modelAngFin.getNumber());
			}
		}
		
		/**
		 * Lo invoca el agente cuando se recibe nueva configuración.
		 * @param nuevaConfig contenido del mensaje
		 */
		public void cambioConfiguracio(String nuevaConfig) {
			//programamos la actualizacion de la ventana
			miGui.labelConfiguración.setText(nuevaConfig);
			SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					labelConfiguración.repaint();						
				}
			});

		}
		
		/**
		 * Lo invoca el agente cuando lanza petición de datos
		 */
		public void comienzaEscaneo() {
			if(cbEscaneo.isSelected()) {
				//estamos en escaneo continuo
				//Marcamos estamos esperando
				jlEsperando.setEnabled(true);
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						jlEsperando.repaint();						
					}
				});
			} else {
				//Fue petición del botón. debemos activar etiquieta y desactivar botón
				jlEsperando.setEnabled(true);
				jbPideBarrido.setEnabled(false);
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						jlEsperando.repaint();
						jbPideBarrido.repaint();
					}
				});
			}
		}
		
		/**
		 * Lo invoca el agente cuando termina la petición de datos.
		 */
		public void finalizaEscaneo() {
			if(cbEscaneo.isSelected()) {
				//estamos en escaneo continuo
				//Desactivamos estamos esperando
				jlEsperando.setEnabled(false);
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						jlEsperando.repaint();						
					}
				});
			} else {
				//Fue petición del botón. debemos activar etiquieta y desactivar botón
				jlEsperando.setEnabled(false);
				jbPideBarrido.setEnabled(true);
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						jlEsperando.repaint();
						jbPideBarrido.repaint();
					}
				});
			}
			
		}
	}
	
	//Clases privadas con los comportamientos
	 
	/**
	 * Comportamiento ciclico para iniciar la peticion del barrido cada cierto tiempo.
	 * 
	 * @author alberto
	 * 
	 */
	private class ComportaEscaneoContinuo extends CyclicBehaviour {
		/** Comportamiento que se reutilizará en cada iteración */
		private ComportaPideBarrido rb=null;
		/** Tiempo de espera entre cada iteración */
		private static final long esperaMs=1000;
		/** Amacena milisegundo en que debe ser la siguiente iteración */
		private long siguiente=0; 

		/**
		 * Añade el comportamiento {@link ComportaPideBarrido} si se ha cumplide el tiempo.
		 * <p>
		 * Se controla que el tiempo de la siguiente iteración se haya alcanzado. 
		 * Si no es así se bloquea por el tiempo restante. 
		 * Esto debe ser así porque el comportamiento se activa cada vez que llega mensaje.
		 */
		public void action() {
			long ct=System.currentTimeMillis();
			//si no ha llegado nuestro tiempo nos bloqueamos otra vez
			if (ct<siguiente) {
				agentLog("Faltan "+(siguiente-ct)+" ms");
				block(siguiente-ct);
				return;
			}

			//Lanzamos petición si tenemos referencia al agente
			if (agenteLMS()!=null) {
				agentLog("Entramos accion PideBarrido");
				if(rb==null) {
					rb=new ComportaPideBarrido();
					addBehaviour(rb);
					agentLog("Añadimos el comportamiento RequestBarrido");
				} else {
					if(rb.done()) {
						//Lanzamos nuevamente
						rb.reset();
						addBehaviour(rb);
					} else {
						agentLog("RequestBarrido no ha terminado, no lo lanzamos");					
					}
				}
			}

			//esperamos según se indica en el GUI
			siguiente=System.currentTimeMillis() + miGui.modelFrecEscaneo.getNumber().longValue();
			block(esperaMs);
		}
		
	}  //The la clase ComportaEscaneoContinuo

	/**
	 * Implementa conversación para la recepción del barrido.
	 * <p>
	 * Si se recibe correctamente barrido lo actualiza en la ventana gráfica
	 * {@link PanelMuestraBarrido}.
	 * 
	 * @author alberto
	 *
	 */
	private class ComportaPideBarrido extends SimpleAchieveREInitiator {
		/** Lleva número de petición actual */
		private int NumPeticion=0; 

		/** Prepara protocolo para realizar <i>QUERY_REF</i> */
		public ComportaPideBarrido() {
			super(AgenteMuestraBarrido.this,new ACLMessage(ACLMessage.QUERY_REF));
		}

		
		/**
		 * Si se resetea indica escaneo finalizado
		 * @see jade.proto.SimpleAchieveREInitiator#reset()
		 */
		@Override
		public void reset() {
			super.reset();
			miGui.finalizaEscaneo(); //por si había algún escaneo pendiente
		}


		@Override
		/**
		 * Contenido de mensaje de petición es <i>Manda imagen</i>.
		 */
		protected ACLMessage prepareRequest(ACLMessage msg) {
			NumPeticion++; //nueva petición
			agentLog(NumPeticion +" Preparamos mensaje a mandar");
			if (msg==null){
//				agentLog("El mensaje es Null :-(((");
				msg=new ACLMessage(ACLMessage.QUERY_REF);
			}
			if(agenteLMS()==null)
				return	null;	//si no hay destinatarios, no mandamos nada
			msg.addReceiver(agenteLMS());
			msg.setOntology("Barrido");
			msg.setProtocol(FIPANames.InteractionProtocol.FIPA_QUERY);
			msg.setContent("(barrido"
					+" :promedios "+miGui.modelPromedios.getNumber()
					+" :angulo-inicial "+miGui.modelAngIni.getNumber()
					+" :angulo-final "+miGui.modelAngFin.getNumber()
					+")"); //el tipo de mensaje y primer parámetro
			agentLog("Enviamos mensaje con contenido "+msg.getContent());
			miGui.comienzaEscaneo();
			return msg;
		}

		@Override
		/**
		 * Actulizamos el GUI con la imagen recibida en el mensaje.
		 * <p>
		 * Solo se reciben mensaje <i>INFORM</i> 
		 */
		protected void handleInform(ACLMessage msg) {
			agentLog(NumPeticion +" recibimos el mensaje con la informacion");
			try {
				Object om= msg.getContentObject();
				agentLog("El objeto recibido es de la clase:" +
						om.getClass().getName());
				if(! BarridoAngular.class.isInstance(om)) {
					agentLog("El objeto No puede ser tratado porque no es "+
							BarridoAngular.class.getName()
					);
				} else {
					miGui.actualizaBarrido((BarridoAngular)om);
					miGui.finalizaEscaneo();
				}
			} catch (UnreadableException e) {
				agentLog("No se pudo leer el objeto en mensaje por "+ e.getMessage());
				e.printStackTrace();				
				miGui.actualizaBarrido(null);
				miGui.finalizaEscaneo();
			}
		}

		/** 
		 * Indicamos que ha habido fallo
		 * @see jade.proto.SimpleAchieveREInitiator#handleFailure(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleFailure(ACLMessage msg) {
			agentLog("("+getClass().getName()+") Mensaje de fallo:"+msg.getContent());
			miGui.actualizaBarrido(null);
			miGui.finalizaEscaneo(); //por si había algún escaneo pendiente
		}

		/**
		 * Indicamos mensaje no entendido
		 * @see jade.proto.SimpleAchieveREInitiator#handleNotUnderstood(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleNotUnderstood(ACLMessage msg) {
			agentLog("("+getClass().getName()+") Nuestro mensaje no fue entendido:"+msg.getContent());
			miGui.actualizaBarrido(null);
			miGui.finalizaEscaneo(); //por si había algún escaneo pendiente
		}

		/**
		 * Indicamos el agente no lo quiere hacer
		 * @see jade.proto.SimpleAchieveREInitiator#handleRefuse(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleRefuse(ACLMessage msg) {
			agentLog("("+getClass().getName()+") El interlocutor no lo quiere hacer:"+msg.getContent());
			miGui.actualizaBarrido(null);
			miGui.finalizaEscaneo(); //por si había algún escaneo pendiente
		}

	}   // de la clase ComportaPideBarrido

	/**
	 * Comportamiento para pedir configuración a agente LMS e indicarla en el GUI
	 * @author alberto
	 *
	 */
	protected class ComportaPideConfiguracion extends SimpleAchieveREInitiator {
		public ComportaPideConfiguracion() {
			super(AgenteMuestraBarrido.this,new ACLMessage(ACLMessage.REQUEST));
		}
		
		@Override
		protected ACLMessage prepareRequest(ACLMessage msg) {
			if(agenteLMS()==null) {
				agentLog("No hay agente LMS al que pedir");
				return null;
			}
			if (msg==null){
				msg=new ACLMessage(ACLMessage.REQUEST);
			}
			agentLog("Pedimos la configuración");
			msg.addReceiver(agenteLMS());
			msg.setOntology("Configuracion");
			msg.setProtocol(FIPANames.InteractionProtocol.FIPA_REQUEST);
			msg.setContent("(configuracion)");
			return msg;
		}

		@Override
		protected void handleInform(ACLMessage msg) {
			agentLog("Recibimos mensaje con contenido:"+msg.getContent());
			miGui.cambioConfiguracio(msg.getContent());
		}

		/** 
		 * Indicamos que ha habido fallo
		 * @see jade.proto.SimpleAchieveREInitiator#handleFailure(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleFailure(ACLMessage msg) {
			agentLog("("+getClass().getName()+") Mensaje de fallo:"+msg.getContent());
		}

		/**
		 * Indicamos mensaje no entendido
		 * @see jade.proto.SimpleAchieveREInitiator#handleNotUnderstood(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleNotUnderstood(ACLMessage msg) {
			agentLog("("+getClass().getName()+") Nuestro mensaje no fue entendido:"+msg.getContent());
		}

		/**
		 * Indicamos el agente no lo quiere hacer
		 * @see jade.proto.SimpleAchieveREInitiator#handleRefuse(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleRefuse(ACLMessage msg) {
			agentLog("("+getClass().getName()+") El interlocutor no lo quiere hacer:"+msg.getContent());
		}
		
	}  // de la clase ComportaPideConfiguracion

	/**
	 * Comportamiento para pedir configuración a agente LMS e indicarla en el GUI
	 * @author alberto
	 *
	 */
	protected class ComportaPideZonas extends SimpleAchieveREInitiator {
		private boolean delConjunto1;
		private byte deLaZona;
		
//		public ComportaPideZonas() {
//			super(AgenteMuestraBarrido.this,new ACLMessage(ACLMessage.REQUEST));
//		}
		
		/**
		 * @param c1 ¿conjunto 1?
		 * @param zona que zona
		 */
		public ComportaPideZonas(boolean c1, byte zona) {
			super(AgenteMuestraBarrido.this,new ACLMessage(ACLMessage.REQUEST));
			delConjunto1=c1;
			deLaZona=zona;
		}

		/**
		 * Reset sin parámetros hace petición a conjunto1 zonaA
		 * @see jade.proto.SimpleAchieveREInitiator#reset()
		 */
		@Override
		public void reset() {
			reset(true,ZonaLMS.ZONA_A);
		}

		/**
		 * Reset que permite pasar el conjunto y la zona a indagar.
		 * @param delConjunto1 si se pide zona del conjuto 1
		 * @param deLaZona de que zona 
		 * @see jade.proto.SimpleAchieveREInitiator#reset()
		 */
		public void reset(boolean delConjunto1, byte deLaZona) {
			super.reset();
			this.delConjunto1=delConjunto1;
			this.deLaZona=deLaZona;
		}



		@Override
		protected ACLMessage prepareRequest(ACLMessage msg) {
			if(agenteLMS()==null) {
				agentLog("No hay agente LMS al que pedir");
				return null;
			}
			if (msg==null){
				msg=new ACLMessage(ACLMessage.REQUEST);
			}
			agentLog("Pedimos la zona");
			msg.addReceiver(agenteLMS());
			msg.setOntology("ZonasLMS");
			msg.setProtocol(FIPANames.InteractionProtocol.FIPA_REQUEST);
			msg.setContent("(zona :conjunto "+(delConjunto1?"1":"2")
					+" :zona "+ (deLaZona==ZonaLMS.ZONA_A?"A":(deLaZona==ZonaLMS.ZONA_B?"B":"C")) +")");
			return msg;
		}

		@Override
		protected void handleInform(ACLMessage msg) {
			agentLog("Recibimos el mensaje con la informacion de la zona");
			try {
				Object om= msg.getContentObject();
				agentLog("El objeto recibido es de la clase:" +
						om.getClass().getName());
				if(! ZonaLMS.class.isInstance(om)) {
					agentLog("El objeto No puede ser tratado porque no es "+
							ZonaLMS.class.getName()
					);
				} else {
					miGui.actualizaZona((ZonaLMS)om);
				}
			} catch (UnreadableException e) {
				agentLog("No se pudo leer el objeto en mensaje por "+ e.getMessage());
				e.printStackTrace();				
			}
		}

		/** 
		 * Indicamos que ha habido fallo
		 * @see jade.proto.SimpleAchieveREInitiator#handleFailure(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleFailure(ACLMessage msg) {
			agentLog("("+getClass().getName()+") Mensaje de fallo:"+msg.getContent());
		}

		/**
		 * Indicamos mensaje no entendido
		 * @see jade.proto.SimpleAchieveREInitiator#handleNotUnderstood(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleNotUnderstood(ACLMessage msg) {
			agentLog("("+getClass().getName()+") Nuestro mensaje no fue entendido:"+msg.getContent());
		}

		/**
		 * Indicamos el agente no lo quiere hacer
		 * @see jade.proto.SimpleAchieveREInitiator#handleRefuse(jade.lang.acl.ACLMessage)
		 */
		@Override
		protected void handleRefuse(ACLMessage msg) {
			agentLog("("+getClass().getName()+") El interlocutor no lo quiere hacer:"+msg.getContent());
		}


	}  // de la clase ComportaPideConfiguracion
}
